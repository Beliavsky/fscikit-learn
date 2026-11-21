program reference_classifier_chain_driver
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf
  implicit none
  type :: fold_plan
    type(index_split), allocatable :: folds(:)
  end type
  type(classifier_chain) :: chain
  type(multi_output_classifier) :: single
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(sgd_classifier) :: sgd
  type(kneighbors_classifier) :: knn
  type(decision_tree_classifier) :: tree
  type(multinomial_nb) :: mnb
  type(bernoulli_nb) :: bnb
  type(complement_nb) :: cnb
  type(fold_plan), allocatable :: plans(:)
  class(*), allocatable :: base
  real(dp), allocatable :: x(:,:),query(:,:),w(:),sw(:),before(:,:),after(:,:),aug(:,:),response(:,:)
  integer, allocatable :: y(:,:),truth(:,:)
  integer :: n,p,m,t,choice,method,kind,cv,seed,weighted,flags,refit,input,output,j,k,a,b,nfolds
  character(1024) :: input_path,output_path
  call get_command_argument(1,input_path); call get_command_argument(2,output_path)
  open(newunit=input,file=trim(input_path),status='old')
  read(input,*) n,p,m,t,choice,method,kind,cv,seed,weighted,flags,refit
  allocate(x(n,p),y(n,t),query(m,p),truth(m,t),w(n),sw(m),plans(t))
  read(input,*) w; read(input,*) sw
  do j=1,n
    read(input,*) x(j,:)
  end do
  do j=1,n
    read(input,*) y(j,:)
  end do
  do j=1,m
    read(input,*) query(j,:)
  end do
  do j=1,m
    read(input,*) truth(j,:)
  end do
  do j=1,t
    read(input,*) nfolds
    allocate(plans(j)%folds(nfolds))
    do k=1,nfolds
      read(input,*) a,b
      allocate(plans(j)%folds(k)%train(a),plans(j)%folds(k)%test(b))
      read(input,*) plans(j)%folds(k)%train
      read(input,*) plans(j)%folds(k)%test
    end do
  end do
  close(input)
  lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  sgd%loss='log_loss'; sgd%learning_rate='constant'; sgd%eta0=.02_dp; sgd%alpha=.001_dp
  sgd%max_iter=4; sgd%tol=-1; sgd%shuffle=.false.
  if(btest(flags,3)) sgd%loss='hinge'
  if(btest(flags,5)) sgd%loss='modified_huber'
  if(btest(flags,4)) sgd%average=3
  knn%n_neighbors=3; knn%weights='distance'
  tree%max_depth=2; tree%min_samples_leaf=4
  mnb%alpha=.4_dp; bnb%alpha=.4_dp; cnb%alpha=.4_dp; cnb%norm=.true.
  select case(choice)
  case(0)
    allocate(base,source=lr)
  case(1)
    allocate(base,source=nb)
  case(2)
    allocate(base,source=sgd)
  case(3)
    allocate(base,source=knn)
  case(4)
    allocate(base,source=tree)
  case(5)
    allocate(base,source=mnb)
  case(6)
    allocate(base,source=bnb)
  case(7)
    allocate(base,source=cnb)
  end select
  call single%fit(x,y(:,:1),base)
  deallocate(base); allocate(base,source=single%estimators_(1)%estimator_)
  before=base_response(base,query,'predict')
  chain%cv=cv; chain%random_state=seed
  if(kind==1) chain%order=[(j,j=t,1,-1)]
  if(kind==2) chain%random_order=.true.
  select case(method)
  case(0)
    chain%chain_method='predict'
  case(1)
    chain%chain_method='predict_proba'
  case(2)
    chain%chain_method='predict_log_proba'
  case(3)
    chain%chain_method='decision_function'
  case(4)
    chain%chain_methods=[character(20) :: 'decision_function','predict_proba','predict']
  end select
  if(kind>=3) then
    if(weighted==1) then
      call chain%fit(x,y,base,w,plans(1)%folds)
    else
      call chain%fit(x,y,base,splits=plans(1)%folds)
    end if
  else
    if(weighted==1) then
      call chain%fit(x,y,base,w)
    else
      call chain%fit(x,y,base)
    end if
  end if
  open(newunit=output,file=trim(output_path),status='replace')
  call snapshot(.false.)
  if(refit==1) then
    chain%cv=0; chain%random_order=.false.
    if(allocated(chain%order)) deallocate(chain%order)
    call chain%fit(x,y(:,:1),base); call snapshot(.true.)
  end if
  after=base_response(base,query,'predict'); call matrix(before); call matrix(after)
  if(choice==4) call model_state(base)
  call matrix(x); call matrix(query)
  do j=1,n
    write(output,*) y(j,:)
  end do
  write(output,'(*(es25.16e3,1x))') w
  close(output)
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      write(output,'(*(es25.16e3,1x))') a(row,:)
    end do
  end subroutine
  subroutine snapshot(is_refit)
    logical, intent(in) :: is_refit
    type(multi_output_classifier) :: fold_model
    integer :: j,k,column,width,nc
    integer, allocatable :: labels(:),predicted(:,:)
    real(dp), allocatable :: r(:,:)
    write(output,*) chain%n_features_in_,chain%n_outputs_
    write(output,*) trim(chain%chain_method_)
    write(output,*) chain%order_
    do j=1,chain%n_outputs_
      write(output,*) size(chain%classes_(j)%values)
      write(output,*) chain%classes_(j)%values
    end do
    predicted=chain%predict(query)
    do j=1,m
      write(output,*) predicted(j,:)
    end do
    if(btest(flags,0)) call matrix(chain%predict_proba(query))
    if(btest(flags,1)) call matrix(chain%decision_function(query))
    if(btest(flags,2)) write(output,'(*(es25.16e3,1x))') &
      chain%score(query,truth(:,:chain%n_outputs_)),chain%score(query,truth(:,:chain%n_outputs_),sw)
    do j=1,chain%n_outputs_
      call model_state(chain%estimators_(j)%estimator_)
    end do
    if(allocated(aug)) deallocate(aug)
    allocate(aug(n,p+chain%n_outputs_)); aug=0; aug(:,:p)=x
    if((cv==0.and.kind<3).or.is_refit) aug(:,p+1:)=real(y(:,chain%order_),dp)
    do j=1,chain%n_outputs_-1
      column=chain%order_(j); width=p+j-1; nfolds=0
      if(.not.is_refit) nfolds=size(plans(column)%folds)
      if(choice==4) write(output,*) nfolds
      do k=1,nfolds
        call fold_model%fit(aug(plans(column)%folds(k)%train,:width), &
          y(plans(column)%folds(k)%train,column:column),base)
        if(choice==4) call model_state(fold_model%estimators_(1)%estimator_)
        r=base_response(fold_model%estimators_(1)%estimator_, &
          aug(plans(column)%folds(k)%test,:width),trim(chain%chain_method_))
        if(chain%chain_method_=='predict_proba'.or.chain%chain_method_=='predict_log_proba') then
          labels=fold_model%classes_(1)%values; nc=0
          block
            integer, allocatable :: global(:)
            global=chain%classes_(j)%values
            do a=1,size(labels)
              if(labels(a)==global(2)) nc=a
            end do
          end block
          if(nc==0) then
            aug(plans(column)%folds(k)%test,p+j)=0
            if(chain%chain_method_=='predict_log_proba') aug(plans(column)%folds(k)%test,p+j)=-huge(1._dp)
          else
            aug(plans(column)%folds(k)%test,p+j)=r(:,nc)
          end if
        else
          nc=1
          if(size(r,2)>1) nc=2
          aug(plans(column)%folds(k)%test,p+j)=r(:,nc)
        end if
      end do
    end do
    call matrix(aug(:,:p+chain%n_outputs_-1))
  end subroutine
  subroutine model_state(model)
    class(*), intent(in) :: model
    integer :: j
    select type(model)
    type is(logistic_regression)
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); write(output,'(*(es25.16e3,1x))') model%intercept_
    type is(gaussian_nb)
      write(output,*) 2,size(model%theta_,1),size(model%theta_,2)
      call matrix(model%theta_); call matrix(model%var_)
      write(output,'(*(es25.16e3,1x))') model%class_prior_
      write(output,'(*(es25.16e3,1x))') model%class_count_
    type is(sgd_classifier)
      write(output,*) 3,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); write(output,'(*(es25.16e3,1x))') model%intercept_,model%t_
    class is(multinomial_nb)
      write(output,*) 4,size(model%feature_log_prob_,1),size(model%feature_log_prob_,2)
      call matrix(model%feature_log_prob_)
      write(output,'(*(es25.16e3,1x))') model%class_log_prior_
    type is(decision_tree_classifier)
      write(output,*) 5,size(model%classes_),model%node_count_
      write(output,*) model%classes_
      do j=1,model%node_count_
        associate(node=>model%nodes_(j))
          write(output,'(5(i0,1x),*(es25.16e3,1x))') &
            node%left,node%right,node%feature,node%n_samples,node%depth,node%threshold,node%impurity,node%probability
        end associate
      end do
    type is(kneighbors_classifier)
      write(output,*) 6,size(model%search_%x_,1),size(model%search_%x_,2)
      call matrix(model%search_%x_); write(output,*) model%y_
    class default
      call require(.false.,'unexpected classifier-chain reference base')
    end select
  end subroutine
  function base_response(model,q,method) result(r)
    class(*), intent(in) :: model
    real(dp), intent(in) :: q(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: r(:,:)
    integer :: i,j
    if(method=='predict') allocate(r(size(q,1),1))
    select type(model)
    type is(logistic_regression)
      if(method=='predict') then
        r(:,1)=real(model%predict(q),dp)
      else if(method=='decision_function') then
        r=model%decision_function(q)
      else
        r=model%predict_proba(q)
      end if
    type is(gaussian_nb)
      if(method=='predict') then
        r(:,1)=real(model%predict(q),dp)
      else if(method=='predict_log_proba') then
        r=model%predict_log_proba(q); return
      else
        r=model%predict_proba(q)
      end if
    type is(sgd_classifier)
      if(method=='predict') then
        r(:,1)=real(model%predict(q),dp)
      else if(method=='decision_function') then
        r=model%decision_function(q)
      else
        r=model%predict_proba(q)
      end if
    type is(kneighbors_classifier)
      if(method=='predict') then
        r(:,1)=real(model%predict(q),dp)
      else
        r=model%predict_proba(q)
      end if
    type is(decision_tree_classifier)
      if(method=='predict') then
        r(:,1)=real(model%predict(q),dp)
      else
        r=model%predict_proba(q)
      end if
    class is(multinomial_nb)
      if(method=='predict') then
        r(:,1)=real(model%predict(q),dp)
      else if(method=='predict_log_proba') then
        r=model%predict_log_proba(q); return
      else
        r=model%predict_proba(q)
      end if
    end select
    if(method=='predict_log_proba') then
      do j=1,size(r,2)
        do i=1,size(r,1)
          if(r(i,j)==0) then
            r(i,j)=ieee_value(0._dp,ieee_negative_inf)
          else
            r(i,j)=log(r(i,j))
          end if
        end do
      end do
    end if
  end function
end program
