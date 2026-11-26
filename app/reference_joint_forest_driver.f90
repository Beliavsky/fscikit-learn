program reference_joint_forest_driver
 use sklearn_common,only:dp
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor
 implicit none
 class(random_forest_regressor),allocatable::model
 real(dp),allocatable::x(:,:),y(:,:),w(:),prediction(:,:),values(:,:),importance(:,:)
 integer,allocatable::samples(:,:),meta(:,:),leaves(:,:)
 integer::n,p,q,mode,seed,nt,boot,ns,oob,weighted,depth,budget,nf,leaf,i,j,at,total,status
 real(dp)::alpha,decrease
 character(16)::criterion
 do
  read(*,*,iostat=status)n,p,q,mode,criterion,seed,nt,boot,ns,oob,weighted,depth,budget,nf,leaf,alpha,decrease
  if(status/=0)exit
  allocate(x(n,p),y(n,q),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,n;read(*,*)y(i,:);end do
  read(*,*)w
  if(mode==0)then
   allocate(random_forest_regressor::model)
  else
   allocate(extra_trees_regressor::model)
  end if
  model%criterion=criterion;model%random_state=seed;model%n_estimators=nt;model%max_samples=ns
  call model%set_bootstrap(boot==1);model%oob_score=oob==1
  model%max_depth=depth;model%max_leaf_nodes=budget;model%max_features=nf;model%min_samples_leaf=leaf
  model%ccp_alpha=alpha;model%min_impurity_decrease=decrease
  if(weighted==1)then
   call model%fit(x,y,w)
  else
   call model%fit(x,y)
  end if
  prediction=model%predict_multi(x);samples=model%estimators_samples();leaves=model%apply(x)
  total=sum(model%estimators_%node_count_);allocate(values(total,q),importance(nt,p),meta(nt,5));at=1
  do i=1,nt
   meta(i,:)=[model%estimators_(i)%node_count_,model%estimators_(i)%get_depth(), &
    model%estimators_(i)%get_n_leaves(),model%estimators_(i)%n_features_in_,model%estimators_(i)%n_outputs_]
   importance(i,:)=model%estimators_(i)%feature_importances_
   do j=1,model%estimators_(i)%node_count_;values(at,:)=model%estimators_(i)%nodes_(j)%values;at=at+1;end do
  end do
  write(*,'(*(i0,1x))')model%n_features_in_,model%n_outputs_,size(model%estimators_)
  write(*,'(*(es26.17e3,1x))')transpose(prediction)
  write(*,'(*(i0,1x))')transpose(leaves)
  write(*,'(*(es26.17e3,1x))')model%feature_importances_
  write(*,'(*(i0,1x))')model%estimators_%random_state
  write(*,'(*(i0,1x))')transpose(samples)
  if(oob==1)then
   if(q==1)then
    write(*,'(*(es26.17e3,1x))')model%oob_prediction_
   else
    write(*,'(*(es26.17e3,1x))')transpose(model%oob_prediction_multi_)
   end if
   write(*,'(*(es26.17e3,1x))')model%oob_score_
   write(*,'(*(i0,1x))')model%oob_counts_
  else
   write(*,*)
   write(*,*)
   write(*,*)
  end if
  write(*,'(*(i0,1x))')transpose(meta)
  write(*,'(*(es26.17e3,1x))')transpose(values)
  call nodes('impurity');call nodes('mass');call nodes('count');call nodes('left');call nodes('right')
  call nodes('feature');call nodes('threshold')
  write(*,'(*(es26.17e3,1x))')transpose(importance)
  if(weighted==1)then
   write(*,'(*(es26.17e3,1x))')model%score(x,y,w)
  else
   write(*,'(*(es26.17e3,1x))')model%score(x,y)
  end if
  deallocate(model,x,y,w,values,importance,meta,samples,leaves,prediction)
 end do
contains
 subroutine nodes(name)
  character(*),intent(in)::name
  real(dp),allocatable::data(:)
  integer::k,l,a
  allocate(data(total));a=1
  do k=1,nt
   do l=1,model%estimators_(k)%node_count_
    associate(node=>model%estimators_(k)%nodes_(l))
     select case(name)
     case('impurity');data(a)=node%impurity
     case('mass');data(a)=node%weighted_n_samples
     case('count');data(a)=node%n_samples
     case('left');data(a)=node%left
     case('right');data(a)=node%right
     case('feature');data(a)=node%feature
     case('threshold');data(a)=node%threshold
     end select
    end associate
    a=a+1
   end do
  end do
  write(*,'(*(es26.17e3,1x))')data
 end subroutine
end program
