program reference_feature_clone_driver
 use sklearn_common,only:dp
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor,random_forest_classifier,extra_trees_classifier
 use sklearn_tree,only:decision_tree_regressor,extra_tree_regressor,decision_tree_classifier,extra_tree_classifier
 use sklearn_search,only:grid_search_cv,param
 use sklearn_model_selection_features,only:select_from_model
 implicit none
 class(*),allocatable::base
 type(grid_search_cv)::search
 type(select_from_model)::selector,prefit
 real(dp),allocatable::x(:,:),yr(:),prob(:,:)
 integer,allocatable::yi(:)
 integer::n,p,family,mode,seed,status,i
 do
  read(*,*,iostat=status)n,p,family,mode,seed
  if(status/=0)exit
  allocate(x(n,p),yr(n),yi(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)yr;read(*,*)yi
  if(mod(family,2)==0)then
   if(mode==0)then
    if(family<2)then
     allocate(random_forest_regressor::base)
    else
     allocate(decision_tree_regressor::base)
    end if
   else
    if(family<2)then
     allocate(extra_trees_regressor::base)
    else
     allocate(extra_tree_regressor::base)
    end if
   end if
  else
   if(mode==0)then
    if(family<2)then
     allocate(random_forest_classifier::base)
    else
     allocate(decision_tree_classifier::base)
    end if
   else
    if(family<2)then
     allocate(extra_trees_classifier::base)
    else
     allocate(extra_tree_classifier::base)
    end if
   end if
  end if
  select type(base)
  class is(random_forest_regressor)
   base%warm_start=.true.;base%n_estimators=5;base%random_state=seed;base%max_depth=3;base%oob_score=.true.
   call base%set_bootstrap(.true.);call base%set_max_features(.7_dp)
   call base%fit(x,0*yr)
  class is(random_forest_classifier)
   base%warm_start=.true.;base%n_estimators=5;base%random_state=seed;base%max_depth=3;base%oob_score=.true.
   call base%set_bootstrap(.true.);call base%set_max_features(.7_dp)
   call base%fit(x,0*yi)
  class is(decision_tree_regressor)
   base%random_state=seed;base%max_depth=3;call base%set_max_features(.7_dp);call base%fit(x,0*yr)
  class is(decision_tree_classifier)
   base%random_state=seed;base%max_depth=3;call base%set_max_features(.7_dp);call base%fit(x,0*yi)
  end select
  search=grid_search_cv();allocate(search%estimator,source=base);search%cv=3
  allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(1))
  search%param_grid(1)%axes(1)%name='max_features'
  search%param_grid(1)%axes(1)%values=[param(1),param(1._dp),param(.5_dp),param('sqrt'),param('log2'),param('none'),param(p+3)]
  selector=select_from_model();prefit=select_from_model();prefit%prefit=.true.
  if(mod(family,2)==0)then
   call search%fit(x,yr);call selector%fit(x,yr,base)
  else
   call search%fit(x,yi);call selector%fit(x,yi,base)
  end if
  call prefit%fit(x,estimator=base)
  write(*,'(*(es26.17e3,1x))')search%cv_results_%mean_test_score
  write(*,'(*(es26.17e3,1x))')search%cv_results_%std_test_score
  write(*,'(*(es26.17e3,1x))')transpose(search%cv_results_%split_test_score(:,:,1))
  write(*,'(*(i0,1x))')search%cv_results_%rank_test_score
  select type(best=>search%best_estimator_)
  class is(random_forest_regressor)
   write(*,'(*(i0,1x))')search%best_index_,size(best%estimators_),merge(1,0,best%bootstrap), &
    merge(1,0,best%warm_start),merge(1,0,allocated(best%oob_counts_))
  class is(random_forest_classifier)
   write(*,'(*(i0,1x))')search%best_index_,size(best%estimators_),merge(1,0,best%bootstrap), &
    merge(1,0,best%warm_start),merge(1,0,allocated(best%oob_counts_))
  class is(decision_tree_regressor)
   write(*,'(*(i0,1x))')search%best_index_,1,0,0,0
  class is(decision_tree_classifier)
   write(*,'(*(i0,1x))')search%best_index_,1,0,0,0
  end select
  if(mod(family,2)==0)then
   write(*,'(*(es26.17e3,1x))')search%predict(x);write(*,*)
  else
   write(*,'(*(i0,1x))')search%predict_class(x);prob=search%predict_proba(x)
   write(*,'(*(es26.17e3,1x))')transpose(prob)
  end if
  write(*,'(*(es26.17e3,1x))')selector%feature_importances()
  write(*,'(*(i0,1x))')merge(1,0,selector%get_support())
  write(*,'(*(es26.17e3,1x))')prefit%feature_importances()
  write(*,'(*(i0,1x))')merge(1,0,prefit%get_support())
  select type(base)
  class is(random_forest_regressor)
   write(*,'(*(es26.17e3,1x))')base%predict(x)
   write(*,'(*(i0,1x))')merge(1,0,allocated(base%oob_counts_)),size(base%estimators_)
  class is(random_forest_classifier)
   write(*,'(*(i0,1x))')base%predict(x)
   write(*,'(*(i0,1x))')merge(1,0,allocated(base%oob_counts_)),size(base%estimators_)
  class is(decision_tree_regressor)
   write(*,'(*(es26.17e3,1x))')base%predict(x)
   write(*,'(*(i0,1x))')0,1
  class is(decision_tree_classifier)
   write(*,'(*(i0,1x))')base%predict(x)
   write(*,'(*(i0,1x))')0,1
  end select
  write(*,'(*(es26.17e3,1x))')search%best_score_
  deallocate(base,x,yr,yi)
 end do
end program
