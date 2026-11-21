program reference_dependence_driver
  use sklearn
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  type(partial_dependence_result) :: result
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(kneighbors_classifier) :: knn
  type(linear_svc) :: margin
  real(dp), allocatable :: train(:,:),target(:),x(:,:),w(:),before(:,:),after(:,:),values(:,:)
  real(dp), allocatable :: coef_before(:,:),intercept_before(:),coef_after(:,:),intercept_after(:)
  integer, allocatable :: features(:),labels(:),categories(:)
  logical, allocatable :: mask(:)
  real(dp) :: limits(2)
  integer :: mode,case_,n,p,m,k,resolution,kind_id,cat_id,weighted,special,input,output,i,j,points,outputs,calls=0
  character(10) :: kind
  character(1024) :: path,outpath
  call get_command_argument(1,path); call get_command_argument(2,outpath)
  open(newunit=input,file=trim(path),status='old')
  read(input,*) mode,case_,n,p,m,k,resolution,kind_id,cat_id,weighted,special
  allocate(train(n,p),target(n),x(m,p),w(m),features(k),labels(n),mask(p),categories(0))
  read(input,*) features; read(input,*) limits; read(input,*) w
  do i=1,n
    read(input,*) train(i,:)
  end do
  read(input,*) target
  do i=1,m
    read(input,*) x(i,:)
  end do
  close(input); labels=int(target)
  rr%alpha=.7_dp; lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  knn%n_neighbors=3; knn%weights='distance'
  margin%c=.7_dp; margin%tol=1.e-10_dp; margin%max_iter=10000
  select case(mode)
  case(1)
    call ols%fit(train,target)
  case(2)
    call rr%fit(train,target)
  case(3:6)
    call lr%fit(train,labels)
  case(7)
    call nb%fit(train,labels)
  case(8)
    call knn%fit(train,labels)
  case(9)
    call margin%fit(train,labels)
  end select
  call parameters(coef_before,intercept_before); before=model_response(x)
  kind='average'
  if(kind_id==1) kind='individual'
  if(kind_id==2) kind='both'
  mask=.false.
  if(cat_id==1) mask(3)=.true.
  if(cat_id==2) categories=[2]
  if(cat_id==3) categories=[100]
  open(newunit=output,file=trim(outpath),status='replace')
  if(weighted/=0) then
    call calculate(w)
  else
    call calculate()
  end if
  points=1
  do j=1,k
    points=points*size(result%grid_values(j)%values)
  end do
  outputs=size(before,2)
  write(output,*) k,outputs,points,merge(1,0,allocated(result%average)),merge(1,0,allocated(result%individual)),calls
  do j=1,k
    write(output,*) size(result%grid_values(j)%values); call vector(result%grid_values(j)%values)
  end do
  if(allocated(result%average)) call matrix(result%average)
  if(allocated(result%individual)) then
    do j=1,outputs
      call matrix(result%individual(j,:,:))
    end do
  end if
  after=model_response(x); call matrix(before); call matrix(after)
  call parameters(coef_after,intercept_after)
  write(output,*) size(coef_before,1),size(coef_before,2)
  call matrix(coef_before); call vector(intercept_before)
  call matrix(coef_after); call vector(intercept_after)
  call matrix(train); call vector(target); call matrix(x); call vector(w)
  close(output)
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    write(output,'(*(es25.16e3,1x))') a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: j
    do j=1,size(a,1)
      call vector(a(j,:))
    end do
  end subroutine
  subroutine calculate(weights)
    real(dp), intent(in), optional :: weights(:)
    select case(cat_id)
    case(0)
      result=partial_dependence(x,features,predict,limits,resolution,kind=trim(kind),sample_weight=weights)
    case(1)
      result=partial_dependence(x,features,predict,limits,resolution,mask,trim(kind),weights)
    case(2:3)
      result=partial_dependence(x,features,predict,limits,resolution,categories,trim(kind),weights)
    case(4)
      result=partial_dependence(x,features,predict,limits,resolution,[2_int64],trim(kind),weights)
    end select
  end subroutine
  subroutine parameters(coef,intercept)
    real(dp), allocatable, intent(out) :: coef(:,:),intercept(:)
    select case(mode)
    case(1)
      coef=reshape(ols%coef_,[1,p]); intercept=[ols%intercept_]
    case(2)
      coef=reshape(rr%coef_,[1,p]); intercept=[rr%intercept_]
    case(3:6)
      coef=lr%coef_; intercept=lr%intercept_
    case(7)
      coef=nb%theta_; intercept=nb%class_prior_
    case(9)
      coef=margin%coef_; intercept=margin%intercept_
    case default
      allocate(coef(0,p),intercept(0))
    end select
  end subroutine
  function model_response(features) result(values)
    real(dp), intent(in) :: features(:,:)
    real(dp), allocatable :: values(:,:),clean(:,:)
    integer :: j
    select case(mode)
    case(0)
      clean=features
      where(ieee_is_nan(clean)) clean=0
      allocate(values(size(features,1),2)); values=0
      do j=1,size(features,2)
        values(:,1)=values(:,1)+j*clean(:,j)
        values(:,2)=values(:,2)+clean(:,j)**2/j
      end do
      values(:,1)=values(:,1)+clean(:,1)*clean(:,size(features,2))
    case(1)
      allocate(values(size(features,1),1)); values(:,1)=ols%predict(features)
    case(2)
      allocate(values(size(features,1),1)); values(:,1)=rr%predict(features)
    case(3)
      values=lr%predict_proba(features)
    case(4)
      clean=lr%predict_proba(features); values=clean(:,2:2)
    case(5:6)
      values=lr%decision_function(features)
    case(7)
      values=nb%predict_proba(features)
    case(8)
      values=knn%predict_proba(features)
    case(9)
      values=margin%decision_function(features)
    end select
    select case(special)
    case(1)
      values(:,size(values,2))=ieee_value(0._dp,ieee_quiet_nan)
    case(2)
      values(:,size(values,2))=ieee_value(0._dp,ieee_positive_inf)
    case(3)
      values(:,size(values,2))=ieee_value(0._dp,ieee_negative_inf)
    case(4)
      values(:,size(values,2))=ieee_value(0._dp,ieee_negative_inf)
      values(1,size(values,2))=ieee_value(0._dp,ieee_positive_inf)
    end select
  end function
  function predict(features) result(values)
    real(dp), intent(in) :: features(:,:)
    real(dp), allocatable :: values(:,:)
    calls=calls+1; values=model_response(features)
    call matrix(features); call matrix(values)
  end function
end program
