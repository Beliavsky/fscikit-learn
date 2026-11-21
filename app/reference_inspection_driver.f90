program reference_inspection_driver
  use sklearn
  use, intrinsic :: iso_fortran_env, only: real32
  use sklearn_common, only: unique_labels,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  type(permutation_importance_result) :: single
  type(permutation_importance_result), allocatable :: results(:)
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(kneighbors_classifier) :: knn
  real(dp), allocatable :: x(:,:),y(:,:),w(:),before(:,:),after(:,:),coef_before(:,:),intercept_before(:)
  integer, allocatable :: labels(:),classes(:)
  integer :: n,p,repeats,seed,count,metrics,weighted,mode,special,input,output,i,k,calls=0,fractional
  real(dp) :: fraction
  character(1024) :: path,outpath
  call get_command_argument(1,path); call get_command_argument(2,outpath)
  open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,repeats,seed,count,metrics,weighted,mode,special,fractional,fraction
  allocate(x(n,p),y(n,2),w(n)); read(input,*) w
  do i=1,n
    read(input,*) x(i,:)
  end do
  do i=1,n
    read(input,*) y(i,:)
  end do
  close(input)
  labels=int(y(:,1)); classes=unique_labels(labels)
  rr%alpha=.7_dp; lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  knn%n_neighbors=3; knn%weights='distance'
  select case(mode)
  case(1)
    call ols%fit(x,y(:,1)); coef_before=reshape(ols%coef_,[1,p]); intercept_before=[ols%intercept_]
  case(2)
    call rr%fit(x,y(:,1)); coef_before=reshape(rr%coef_,[1,p]); intercept_before=[rr%intercept_]
  case(3)
    call lr%fit(x,labels); coef_before=lr%coef_; intercept_before=lr%intercept_
  case(4)
    call nb%fit(x,labels); coef_before=nb%theta_; intercept_before=nb%class_prior_
  end select
  if(mode==5) call knn%fit(x,labels)
  before=model_response(x)
  open(newunit=output,file=trim(outpath),status='replace')
  if(fractional==2) then
    if(weighted/=0) then
      call calculate(real(fraction,real32),w)
    else
      call calculate(real(fraction,real32))
    end if
  else if(fractional/=0) then
    if(weighted/=0) then
      call calculate(fraction,w)
    else
      call calculate(fraction)
    end if
  else if(count/= -1) then
    if(weighted/=0) then
      call calculate(count,w)
    else
      call calculate(count)
    end if
  else
    if(weighted/=0) then
      call calculate(weights=w)
    else
      call calculate()
    end if
  end if
  write(output,*) size(results),p,repeats,calls
  do k=1,size(results)
    call vector(results(k)%importances_mean); call vector(results(k)%importances_std)
    call matrix(results(k)%importances)
  end do
  after=model_response(x); call matrix(before); call matrix(after)
  if(allocated(coef_before)) then
    call matrix(coef_before); call vector(intercept_before)
    select case(mode)
    case(1)
      call matrix(reshape(ols%coef_,[1,p])); call vector([ols%intercept_])
    case(2)
      call matrix(reshape(rr%coef_,[1,p])); call vector([rr%intercept_])
    case(3)
      call matrix(lr%coef_); call vector(lr%intercept_)
    case(4)
      call matrix(nb%theta_); call vector(nb%class_prior_)
    end select
  end if
  call matrix(x); call matrix(y); call vector(w)
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
  subroutine calculate(max_samples,weights)
    class(*), intent(in), optional :: max_samples
    real(dp), intent(in), optional :: weights(:)
    if(metrics==1) then
      single=permutation_importance(x,scalar_score,repeats,seed,weights,max_samples)
      allocate(results(1)); results(1)=single
    else
      results=permutation_importance_multi(x,score,repeats,seed,weights,max_samples)
    end if
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
    case(1)
      allocate(values(size(features,1),1)); values(:,1)=ols%predict(features)
    case(2)
      allocate(values(size(features,1),1)); values(:,1)=rr%predict(features)
    case(3)
      values=lr%predict_proba(features)
    case(4)
      values=nb%predict_proba(features)
    case(5)
      values=knn%predict_proba(features)
    end select
  end function
  function scalar_score(features,rows,sample_weight) result(value)
    real(dp), intent(in) :: features(:,:)
    integer, intent(in) :: rows(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: value
    real(dp), allocatable :: values(:)
    values=score(features,rows,sample_weight); value=values(1)
  end function
  function score(features,rows,sample_weight) result(values)
    real(dp), intent(in) :: features(:,:)
    integer, intent(in) :: rows(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: values(:),prediction(:,:),errors(:,:),point(:,:),weight(:)
    integer, allocatable :: predicted(:)
    integer :: j
    calls=calls+1; allocate(values(3)); values=[-1._dp,-2._dp,-3._dp]
    if(size(rows)>0) then
      prediction=model_response(features)
      allocate(point(size(rows),3),weight(size(rows))); weight=1
      if(present(sample_weight)) weight=sample_weight
      if(mode<=2) then
        errors=prediction-y(rows,:size(prediction,2))
        point(:,1)=-sum(errors**2,dim=2); point(:,2)=-sum(abs(errors),dim=2)
        point(:,3)=sum(prediction,dim=2)
      else
        predicted=classes([(argmax(prediction(j,:)),j=1,size(rows))])
        point(:,1)=merge(1._dp,0._dp,predicted==labels(rows)); point(:,2)=0
        do j=1,size(classes)
          point(:,2)=point(:,2)-(prediction(:,j)-merge(1._dp,0._dp,labels(rows)==classes(j)))**2
        end do
        point(:,3)=prediction(:,size(classes))
      end if
      do j=1,3
        values(j)=sum(point(:,j)*weight)/sum(weight)
      end do
    end if
    select case(special)
    case(1)
      values(3)=ieee_value(0._dp,ieee_quiet_nan)
    case(2)
      values(3)=ieee_value(0._dp,ieee_positive_inf)
    case(3)
      values(3)=ieee_value(0._dp,ieee_negative_inf)
    case(4)
      values(1)=.75_dp*huge(1._dp)
      if(calls>1) values(1)=-values(1)
      values(2)=.75_dp*huge(1._dp)
      if(mod(calls,2)==0) values(2)=0
      values(3)=0
      if(calls==1) values(3)=ieee_value(0._dp,ieee_negative_inf)
    end select
    write(output,*) size(rows),merge(1,0,present(sample_weight))
    write(output,*) rows
    if(present(sample_weight)) call vector(sample_weight)
    call matrix(features); call vector(values)
    if(metrics==1) values=values(:1)
  end function
  integer function argmax(a) result(index_)
    real(dp), intent(in) :: a(:)
    integer :: j
    index_=1
    do j=2,size(a)
      if(ieee_is_nan(a(index_))) return
      if(ieee_is_nan(a(j))) then
        index_=j; return
      end if
      if(a(j)>a(index_)) index_=j
    end do
  end function
end program
