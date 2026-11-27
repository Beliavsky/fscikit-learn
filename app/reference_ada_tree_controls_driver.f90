program reference_ada_tree_controls_driver
 use sklearn_common,only:dp
 use sklearn_adaboost,only:adaboost_classifier
 use sklearn_adaboost_regression,only:adaboost_regressor
 use sklearn_tree,only:decision_tree_classifier,decision_tree_regressor
 use sklearn_parameter_space,only:parameter_set,param
 use sklearn_search_adapters,only:set_search_params
 implicit none
 type(adaboost_classifier)::cls
 type(adaboost_regressor)::reg
 type(parameter_set)::params
 real(dp),allocatable::x(:,:),query(:,:),y(:),w(:),stages(:,:),values(:)
 integer,allocatable::labels(:)
 integer::kind,n,p,seed,nt,depth,budget,weighted,status,i,j,k,t,nf,number
 real(dp)::ccp,fraction,decrease,rate,f
 character(64)::leaf_token,split_token,feature_token
 character(16)::criterion,splitter,loss
 allocate(params%items(14))
 do
  read(*,*,iostat=status)kind,n,p,seed,nt,depth,criterion,splitter,leaf_token,split_token,feature_token, &
   budget,ccp,fraction,decrease,rate,loss,weighted
  if(status/=0)exit
  allocate(x(n,p),y(n),w(n),labels(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)y;read(*,*)w;labels=int(y);query=x+.137_dp
  params%items(1)%name='n_estimators';params%items(1)%value=param(nt)
  params%items(2)%name='random_state';params%items(2)%value=param(seed)
  params%items(3)%name='estimator__max_depth';params%items(3)%value=param(depth)
  params%items(4)%name='estimator__criterion';params%items(4)%value=param(trim(criterion))
  params%items(5)%name='estimator__splitter';params%items(5)%value=param(trim(splitter))
  params%items(6)%name='estimator__min_samples_leaf';call token_value(leaf_token,6)
  params%items(7)%name='estimator__min_samples_split';call token_value(split_token,7)
  params%items(8)%name='estimator__max_features';call token_value(feature_token,8)
  params%items(9)%name='estimator__max_leaf_nodes';params%items(9)%value=param(budget)
  params%items(10)%name='estimator__ccp_alpha';params%items(10)%value=param(ccp)
  params%items(11)%name='estimator__min_weight_fraction_leaf';params%items(11)%value=param(fraction)
  params%items(12)%name='estimator__min_impurity_decrease';params%items(12)%value=param(decrease)
  params%items(13)%name='learning_rate';params%items(13)%value=param(rate)
  params%items(14)%name='estimator__random_state';params%items(14)%value=param(91)
  if(kind==0)then
   reg=adaboost_regressor();call set_search_params(reg,params);reg%loss=loss
   if(weighted==1)then
    call reg%fit(x,y,w)
   else
    call reg%fit(x,y)
   end if
   nt=reg%n_estimators_
   write(*,'(*(i0,1x))')reg%n_features_in_,nt,0
   call vector(reg%estimator_weights_);call vector(reg%estimator_errors_);call vector(reg%feature_importances_)
   call vector(reg%predict(x));call vector(reg%predict(query))
   stages=reg%staged_predict(query);call vector(reshape(transpose(stages),[size(stages)]))
   write(*,'(*(es26.17e3,1x))')reg%score(x,y),reg%score(x,y,w)
   write(*,'(*(es26.17e3,1x))')reg%staged_score(x,y),reg%staged_score(x,y,w)
   write(*,'(*(i0,1x))')(reg%estimators_(t)%node_count_,reg%estimators_(t)%get_depth(), &
    reg%estimators_(t)%get_n_leaves(),reg%estimators_(t)%n_features_in_,reg%estimators_(t)%n_outputs_, &
    reg%estimators_(t)%max_features_,t=1,nt)
   write(*,'(*(i0,1x))')reg%estimators_%random_state
   write(*,'(*(es26.17e3,1x))')((reg%estimators_(t)%nodes_(j)%value,j=1,reg%estimators_(t)%node_count_),t=1,nt)
   call reg_nodes('impurity');call reg_nodes('mass');call reg_nodes('count');call reg_nodes('left');call reg_nodes('right')
   call reg_nodes('feature');call reg_nodes('threshold')
   write(*,'(*(es26.17e3,1x))')(reg%estimators_(t)%feature_importances_,t=1,nt)
   write(*,*)
   write(*,*)
   write(*,'(*(i0,1x))')reg%estimator_%node_count_,reg%estimator_%n_features_in_,reg%estimator_%random_state
  else
   cls=adaboost_classifier();call set_search_params(cls,params)
   if(weighted==1)then
    call cls%fit(x,labels,w)
   else
    call cls%fit(x,labels)
   end if
   nt=cls%n_estimators_
   write(*,'(*(i0,1x))')cls%n_features_in_,nt,cls%n_classes_
   call vector(cls%estimator_weights_);call vector(cls%estimator_errors_);call vector(cls%feature_importances_)
   write(*,'(*(i0,1x))')cls%predict(x)
   write(*,'(*(i0,1x))')cls%predict(query)
   write(*,'(*(i0,1x))')transpose(cls%staged_predict(query))
   write(*,'(*(es26.17e3,1x))')cls%score(x,labels),cls%score(x,labels,w)
   write(*,'(*(es26.17e3,1x))')cls%staged_score(x,labels),cls%staged_score(x,labels,w)
   write(*,'(*(i0,1x))')(cls%estimators_(t)%node_count_,cls%estimators_(t)%get_depth(), &
    cls%estimators_(t)%get_n_leaves(),cls%estimators_(t)%n_features_in_,cls%estimators_(t)%n_outputs_, &
    cls%estimators_(t)%max_features_,t=1,nt)
   write(*,'(*(i0,1x))')cls%estimators_%random_state
   write(*,'(*(es26.17e3,1x))')((cls%estimators_(t)%nodes_(j)%probability,j=1,cls%estimators_(t)%node_count_),t=1,nt)
   call cls_nodes('impurity');call cls_nodes('mass');call cls_nodes('count');call cls_nodes('left');call cls_nodes('right')
   call cls_nodes('feature');call cls_nodes('threshold')
   write(*,'(*(es26.17e3,1x))')(cls%estimators_(t)%feature_importances_,t=1,nt)
   stages=cls%decision_function(query);call vector(reshape(transpose(stages),[size(stages)]))
   stages=cls%predict_proba(query);call vector(reshape(transpose(stages),[size(stages)]))
   write(*,'(*(i0,1x))')cls%estimator_%node_count_,cls%estimator_%n_features_in_,cls%estimator_%random_state
  end if
  deallocate(x,y,w,labels,query)
 end do
contains
 subroutine vector(a)
  real(dp),intent(in)::a(:)
  write(*,'(*(es26.17e3,1x))')a
 end subroutine
 subroutine token_value(token,index)
  character(*),intent(in)::token
  integer,intent(in)::index
  select case(trim(token))
  case('none','sqrt','log2');params%items(index)%value=param(trim(token))
  case default
   if(scan(trim(token),'.eEdD')>0)then
    read(token,*)f;params%items(index)%value=param(f)
   else
    read(token,*)number;params%items(index)%value=param(number)
   end if
  end select
 end subroutine
 subroutine reg_nodes(name)
  character(*),intent(in)::name
  type(decision_tree_regressor)::tree
  do t=1,nt
   tree=reg%estimators_(t)
   do j=1,tree%node_count_
    select case(name)
    case('impurity');write(*,'(es26.17e3,1x)',advance='no')tree%nodes_(j)%impurity
    case('mass');write(*,'(es26.17e3,1x)',advance='no')tree%nodes_(j)%weighted_n_samples
    case('count');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%n_samples
    case('left');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%left
    case('right');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%right
    case('feature');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%feature
    case('threshold');write(*,'(es26.17e3,1x)',advance='no')tree%nodes_(j)%threshold
    end select
   end do
  end do
  write(*,*)
 end subroutine
 subroutine cls_nodes(name)
  character(*),intent(in)::name
  type(decision_tree_classifier)::tree
  do t=1,nt
   tree=cls%estimators_(t)
   do j=1,tree%node_count_
    select case(name)
    case('impurity');write(*,'(es26.17e3,1x)',advance='no')tree%nodes_(j)%impurity
    case('mass');write(*,'(es26.17e3,1x)',advance='no')tree%nodes_(j)%weighted_n_samples
    case('count');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%n_samples
    case('left');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%left
    case('right');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%right
    case('feature');write(*,'(i0,1x)',advance='no')tree%nodes_(j)%feature
    case('threshold');write(*,'(es26.17e3,1x)',advance='no')tree%nodes_(j)%threshold
    end select
   end do
  end do
  write(*,*)
 end subroutine
end program
