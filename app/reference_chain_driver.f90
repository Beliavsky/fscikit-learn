program reference_chain_driver
  use sklearn
  implicit none
  type(regressor_chain) :: chain
  type(linear_regression) :: linear
  type(ridge) :: regularized
  type(elastic_net) :: elastic
  type(sgd_regressor) :: sgd
  type(kneighbors_regressor) :: knn
  type(decision_tree_regressor) :: tree
  type(index_split), allocatable :: folds(:)
  class(*), allocatable :: base
  real(dp), allocatable :: x(:,:),y(:,:),query(:,:),truth(:,:),w(:),sw(:),prediction(:,:),before(:),after(:)
  integer, allocatable :: ids(:)
  integer :: n,p,m,t,kind,choice,cv,seed,weighted,refit,input,output,j,k
  character(1024) :: input_path,output_path
  call get_command_argument(1,input_path); call get_command_argument(2,output_path)
  open(newunit=input,file=trim(input_path),status='old')
  read(input,*) n,p,m,t,choice,kind,cv,seed,weighted,refit
  allocate(x(n,p),y(n,t),query(m,p),truth(m,t),w(n),sw(m))
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
  close(input)
  regularized%alpha=.7_dp
  elastic%alpha=.03_dp; elastic%l1_ratio=.4_dp; elastic%tol=1.e-12_dp; elastic%max_iter=20000
  sgd%learning_rate='constant'; sgd%eta0=.02_dp; sgd%alpha=.001_dp
  sgd%max_iter=4; sgd%tol=-1; sgd%shuffle=.false.
  knn%n_neighbors=3; knn%weights='distance'
  tree%max_depth=2; tree%min_samples_leaf=5
  select case(choice)
  case(0)
    allocate(base,source=linear)
  case(1)
    allocate(base,source=regularized)
  case(2)
    allocate(base,source=elastic)
  case(3)
    allocate(base,source=sgd)
  case(4)
    allocate(base,source=knn)
  case(5)
    allocate(base,source=tree)
  end select
  call fit_base(base,x,y(:,1)); before=predict_base(base,query)
  chain%cv=cv; chain%random_state=seed
  if(kind==1) chain%order=[(j,j=t,1,-1)]
  if(kind==2) chain%random_order=.true.
  if(kind>=3) then
    allocate(folds(3)); ids=[(j,j=1,n)]
    do k=1,3
      folds(k)%test=pack(ids,mod(ids-1,3)==k-1)
      folds(k)%train=pack(ids,mod(ids-1,3)/=k-1)
      if(kind==4) folds(k)%train=ids
      if(kind==5) folds(k)%train=pack(ids,mod(ids-1,3)/=k-1.and.mod(ids,2)==0)
      if(kind==6) folds(k)%train=[folds(k)%train,folds(k)%train(:2)]
    end do
  end if
  if(kind>=3) then
    if(weighted==1) then
      call chain%fit(x,y,base,w,folds)
    else
      call chain%fit(x,y,base,splits=folds)
    end if
  else
    if(weighted==1) then
      call chain%fit(x,y,base,w)
    else
      call chain%fit(x,y,base)
    end if
  end if
  open(newunit=output,file=trim(output_path),status='replace')
  call snapshot()
  if(refit==1) then
    chain%cv=0; chain%random_order=.false.
    if(allocated(chain%order)) deallocate(chain%order)
    call chain%fit(x,y(:,:1),base); call snapshot()
  end if
  after=predict_base(base,query)
  write(output,'(*(es25.16e3,1x))') before
  write(output,'(*(es25.16e3,1x))') after
  if(choice==5) call write_tree(base)
  call matrix(x); call matrix(y); call matrix(query)
  write(output,'(*(es25.16e3,1x))') w
  close(output)
contains
  subroutine snapshot()
    integer :: j
    write(output,*) chain%n_features_in_,chain%n_outputs_
    write(output,*) chain%order_
    prediction=chain%predict(query); call matrix(prediction)
    write(output,'(*(es25.16e3,1x))') chain%score(query,truth(:,:chain%n_outputs_)), &
      chain%score(query,truth(:,:chain%n_outputs_),sw)
    do j=1,chain%n_outputs_
      select type(model=>chain%estimators_(j)%estimator_)
      class is(linear_regression)
        write(output,*) 1
        write(output,'(*(es25.16e3,1x))') model%coef_,model%intercept_
      type is(sgd_regressor)
        write(output,*) 1
        write(output,'(*(es25.16e3,1x))') model%coef_,model%intercept_,model%t_
      class default
        write(output,*) 0
      end select
    end do
    if(choice==5) call tree_witness()
    if(choice==3) call sgd_witness()
  end subroutine
  subroutine sgd_witness()
    type(sgd_regressor) :: fold_model
    type(index_split), allocatable :: current_folds(:)
    type(kfold) :: splitter
    real(dp), allocatable :: augmented(:,:)
    integer :: j,k,width
    logical :: use_cv
    use_cv=chain%cv/=0.or.(kind>=3.and.chain%n_outputs_>1)
    allocate(augmented(n,p+chain%n_outputs_)); augmented=0; augmented(:,:p)=x
    if(.not.use_cv) augmented(:,p+1:)=y(:,chain%order_)
    if(use_cv.and.chain%n_outputs_>1) then
      if(kind>=3) then
        current_folds=folds
      else
        splitter%n_splits=chain%cv; current_folds=splitter%split(n)
      end if
    end if
    do j=1,chain%n_outputs_-1
      width=p+j-1
      if(use_cv) then
        do k=1,size(current_folds)
          fold_model=sgd
          call fold_model%fit(augmented(current_folds(k)%train,:width),y(current_folds(k)%train,chain%order_(j)))
          augmented(current_folds(k)%test,p+j)=fold_model%predict(augmented(current_folds(k)%test,:width))
        end do
      end if
    end do
    call matrix(augmented(:,:p+chain%n_outputs_-1))
  end subroutine
  subroutine tree_witness()
    type(decision_tree_regressor) :: fold_tree
    type(index_split), allocatable :: current_folds(:)
    type(kfold) :: splitter
    real(dp), allocatable :: augmented(:,:)
    integer :: j,k,width
    logical :: use_cv
    use_cv=chain%cv/=0.or.(kind>=3.and.refit==0).or.(kind>=3.and.chain%n_outputs_>1)
    allocate(augmented(n,p+chain%n_outputs_)); augmented=0; augmented(:,:p)=x
    if(.not.use_cv) augmented(:,p+1:)=y(:,chain%order_)
    if(use_cv.and.chain%n_outputs_>1) then
      if(kind>=3) then
        current_folds=folds
      else
        splitter%n_splits=chain%cv; current_folds=splitter%split(n)
      end if
    end if
    do j=1,chain%n_outputs_
      width=p+j-1
      call write_tree(chain%estimators_(j)%estimator_)
      if(use_cv.and.j<chain%n_outputs_) then
        write(output,*) size(current_folds)
        do k=1,size(current_folds)
          fold_tree=tree
          call fold_tree%fit(augmented(current_folds(k)%train,:width),y(current_folds(k)%train,chain%order_(j)))
          call write_tree(fold_tree)
          augmented(current_folds(k)%test,p+j)=fold_tree%predict(augmented(current_folds(k)%test,:width))
        end do
      else
        write(output,*) 0
      end if
    end do
  end subroutine
  subroutine write_tree(model)
    class(*), intent(in) :: model
    integer :: r
    select type(model)
    type is(decision_tree_regressor)
      write(output,*) model%node_count_
      do r=1,model%node_count_
        associate(node=>model%nodes_(r))
          write(output,'(5(i0,1x),3(es25.16e3,1x))') &
            node%left,node%right,node%feature,node%n_samples,node%depth,node%threshold,node%value,node%impurity
        end associate
      end do
    end select
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: j
    do j=1,size(a,1)
      write(output,'(*(es25.16e3,1x))') a(j,:)
    end do
  end subroutine
  subroutine fit_base(model,x,y)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:)
    select type(model)
    class is(linear_regression)
      call model%fit(x,y)
    type is(sgd_regressor)
      call model%fit(x,y)
    type is(kneighbors_regressor)
      call model%fit(x,y)
    type is(decision_tree_regressor)
      call model%fit(x,y)
    end select
  end subroutine
  function predict_base(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    select type(model)
    class is(linear_regression)
      y=model%predict(x)
    type is(sgd_regressor)
      y=model%predict(x)
    type is(kneighbors_regressor)
      y=model%predict(x)
    type is(decision_tree_regressor)
      y=model%predict(x)
    end select
  end function
end program
