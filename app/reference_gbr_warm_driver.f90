program reference_gbr_warm_driver
 use sklearn_common,only:dp
 use sklearn_ensemble,only:gradient_boosting_regressor
 implicit none
 type(gradient_boosting_regressor)::model
 real(dp),allocatable::x(:,:),query(:,:),y(:),w(:)
 character(16)::loss,criterion
 character(64)::token
 real(dp)::subsample,rate,alpha,decrease,ccp,fraction,value
 integer::n,p,seed,nt,depth,leaf,split,budget,weighted,zero,status,i,j,nf
 integer::warm,fresh,score_ok
 do
  read(*,*,iostat=status)n,p,loss,criterion,seed,nt,depth,token,leaf,split,budget,rate,alpha,decrease,ccp,fraction,weighted,zero,subsample,warm,fresh,score_ok
  if(status/=0)exit
  allocate(x(n,p),y(n),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)y;read(*,*)w;query=x+.137_dp
  if(fresh==1)model=gradient_boosting_regressor()
  model%warm_start=warm==1
  model%subsample=subsample
  model%loss=loss;model%criterion=criterion;model%random_state=seed;model%n_estimators=nt
  model%max_depth=depth;model%min_samples_leaf=leaf;model%min_samples_split=split;model%max_leaf_nodes=budget
  model%learning_rate=rate;model%alpha=alpha;model%min_impurity_decrease=decrease;model%ccp_alpha=ccp
  model%min_weight_fraction_leaf=fraction;model%init_zero=zero==1
  select case(trim(token))
  case('none','sqrt','log2');call model%set_max_features(trim(token))
  case default
   if(scan(trim(token),'.eEdD')>0)then
    read(token,*)value;call model%set_max_features(value)
   else
    read(token,*)nf;call model%set_max_features(nf)
   end if
  end select
  if(weighted==1)then
   call model%fit(x,y,w)
  else
   call model%fit(x,y)
  end if
  write(*,'(*(i0,1x))')model%n_features_in_,model%n_estimators_,model%n_trees_per_iteration_,model%max_features_
  write(*,'(*(es26.17e3,1x))')model%init_
  write(*,'(*(es26.17e3,1x))')model%train_score_
  write(*,'(*(es26.17e3,1x))')model%feature_importances_
  write(*,'(*(es26.17e3,1x))')model%predict(x)
  write(*,'(*(es26.17e3,1x))')model%predict(query)
  write(*,'(*(es26.17e3,1x))')transpose(model%staged_predict(query))
  write(*,'(*(i0,1x))')transpose(model%apply(query))
  write(*,'(*(i0,1x))')(model%estimators_(i)%node_count_,model%estimators_(i)%get_depth(), &
   model%estimators_(i)%get_n_leaves(),model%estimators_(i)%max_features_,i=1,nt)
  write(*,'(*(es26.17e3,1x))')((model%estimators_(i)%nodes_(j)%value,j=1,model%estimators_(i)%node_count_),i=1,nt)
  call nodes('impurity');call nodes('mass');call nodes('count');call nodes('left');call nodes('right')
  call nodes('feature');call nodes('threshold')
  write(*,'(*(es26.17e3,1x))')(model%estimators_(i)%feature_importances_,i=1,nt)
  write(*,'(*(es26.17e3,1x))')model%huber_delta_
  if(score_ok==1)then
  if(weighted==1)then
   write(*,'(*(es26.17e3,1x))')model%score(x,y,w)
  else
   write(*,'(*(es26.17e3,1x))')model%score(x,y)
  end if
  else
   write(*,*)
  end if
  if(allocated(model%oob_scores_))then
   write(*,'(*(es26.17e3,1x))')model%oob_scores_
   write(*,'(*(es26.17e3,1x))')model%oob_improvement_
  else
   write(*,*)
   write(*,*)
  end if
  write(*,'(*(es26.17e3,1x))')model%oob_score_
  deallocate(x,y,w,query)
 end do
contains
 subroutine nodes(name)
  character(*),intent(in)::name
  real(dp),allocatable::a(:)
  integer::k,l,at
  allocate(a(sum(model%estimators_%node_count_)));at=0
  do k=1,size(model%estimators_)
   do l=1,model%estimators_(k)%node_count_
    at=at+1
    select case(name)
    case('impurity');a(at)=model%estimators_(k)%nodes_(l)%impurity
    case('mass');a(at)=model%estimators_(k)%nodes_(l)%weighted_n_samples
    case('count');a(at)=model%estimators_(k)%nodes_(l)%n_samples
    case('left');a(at)=model%estimators_(k)%nodes_(l)%left
    case('right');a(at)=model%estimators_(k)%nodes_(l)%right
    case('feature');a(at)=model%estimators_(k)%nodes_(l)%feature
    case('threshold');a(at)=merge(model%estimators_(k)%nodes_(l)%threshold,0._dp,model%estimators_(k)%nodes_(l)%left/=0)
    end select
   end do
  end do
  if(any(name==[character(16)::'count','left','right','feature']))then
   write(*,'(*(i0,1x))')int(a)
  else
   write(*,'(*(es26.17e3,1x))')a
  end if
 end subroutine
end program
