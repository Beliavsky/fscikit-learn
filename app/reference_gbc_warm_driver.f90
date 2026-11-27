program reference_gbc_warm_driver
 use sklearn_common,only:dp
 use sklearn_ensemble,only:gradient_boosting_classifier
 implicit none
 type(gradient_boosting_classifier)::model
 real(dp),allocatable::x(:,:),query(:,:),stages(:,:,:),w(:)
 integer,allocatable::y(:),leaves(:,:,:)
 integer::n,p,seed,nt,depth,leaf,split,zero,nf,q,i,j,k,t,status,weighted,budget
 real(dp)::subsample,rate,decrease,value,ccp,fraction
 character(16)::loss,criterion
 character(64)::token
 integer::warm,fresh,score_ok
 do
  read(*,*,iostat=status)n,p,loss,criterion,seed,nt,depth,token,leaf,split,rate,decrease,zero,weighted,budget,ccp,fraction,subsample,warm,fresh,score_ok
  if(status/=0)exit
  allocate(x(n,p),y(n),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)y;read(*,*)w;query=x+.137_dp
  if(fresh==1)model=gradient_boosting_classifier()
  model%warm_start=warm==1
  model%subsample=subsample
  model%loss=loss;model%criterion=criterion;model%random_state=seed;model%n_estimators=nt
  model%max_depth=depth;model%min_samples_leaf=leaf;model%min_samples_split=split
  model%learning_rate=rate;model%min_impurity_decrease=decrease;model%init_zero=zero==1
  select case(trim(token))
  case('none','sqrt','log2');call model%set_max_features(trim(token))
  case default
   if(scan(trim(token),'.eEdD')>0)then
    read(token,*)value;call model%set_max_features(value)
   else
    read(token,*)nf;call model%set_max_features(nf)
   end if
  end select
  model%max_leaf_nodes=budget;model%ccp_alpha=ccp;model%min_weight_fraction_leaf=fraction
  if(weighted==1)then
   call model%fit(x,y,w)
  else
   call model%fit(x,y)
  end if
  q=model%n_trees_per_iteration_;stages=model%staged_decision_function(query);leaves=model%apply(query)
  write(*,'(*(i0,1x))')model%n_features_in_,model%n_estimators_,q,model%max_features_,model%n_classes_
  write(*,'(*(i0,1x))')model%classes_
  write(*,'(*(es26.17e3,1x))')model%class_prior_
  write(*,'(*(es26.17e3,1x))')model%init_
  write(*,'(*(es26.17e3,1x))')model%train_score_
  write(*,'(*(es26.17e3,1x))')model%feature_importances_
  write(*,'(*(es26.17e3,1x))')transpose(model%decision_function(x))
  write(*,'(*(es26.17e3,1x))')transpose(model%decision_function(query))
  write(*,'(*(es26.17e3,1x))')transpose(model%predict_proba(query))
  write(*,'(*(i0,1x))')model%predict(query)
  write(*,'(*(es26.17e3,1x))')(((stages(i,k,t),k=1,q),t=1,nt),i=1,n)
  write(*,'(*(i0,1x))')(((leaves(i,t,k),k=1,q),t=1,nt),i=1,n)
  write(*,'(*(i0,1x))')((model%estimators_(t,k)%node_count_,model%estimators_(t,k)%get_depth(), &
   model%estimators_(t,k)%get_n_leaves(),model%estimators_(t,k)%max_features_,k=1,q),t=1,nt)
  write(*,'(*(es26.17e3,1x))')(((model%estimators_(t,k)%nodes_(j)%value,j=1,model%estimators_(t,k)%node_count_),k=1,q),t=1,nt)
  call nodes('impurity');call nodes('mass');call nodes('count');call nodes('left');call nodes('right')
  call nodes('feature');call nodes('threshold')
  write(*,'(*(es26.17e3,1x))')((model%estimators_(t,k)%feature_importances_,k=1,q),t=1,nt)
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
  deallocate(x,y,query,stages,leaves,w)
 end do
contains
 subroutine nodes(name)
  character(*),intent(in)::name
  real(dp),allocatable::a(:)
  integer::at,t,k,j,total
  total=0
  do t=1,nt
   do k=1,q;total=total+model%estimators_(t,k)%node_count_;end do
  end do
  allocate(a(total));at=0
  do t=1,nt
   do k=1,q
    do j=1,model%estimators_(t,k)%node_count_
     at=at+1
     select case(name)
     case('impurity');a(at)=model%estimators_(t,k)%nodes_(j)%impurity
     case('mass');a(at)=model%estimators_(t,k)%nodes_(j)%weighted_n_samples
     case('count');a(at)=model%estimators_(t,k)%nodes_(j)%n_samples
     case('left');a(at)=model%estimators_(t,k)%nodes_(j)%left
     case('right');a(at)=model%estimators_(t,k)%nodes_(j)%right
     case('feature');a(at)=model%estimators_(t,k)%nodes_(j)%feature
     case('threshold');a(at)=merge(model%estimators_(t,k)%nodes_(j)%threshold,0._dp,model%estimators_(t,k)%nodes_(j)%left/=0)
     end select
    end do
   end do
  end do
  if(any(name==[character(16)::'count','left','right','feature']))then
   write(*,'(*(i0,1x))')int(a)
  else
   write(*,'(*(es26.17e3,1x))')a
  end if
 end subroutine
end program
