! Native adaptation of sklearn 1.6.1 feature_selection/_sequential.py (BSD-3-Clause).
module sklearn_sequential_selection
  use sklearn_common
  use sklearn_linear_model, only: linear_regression
  use sklearn_model_selection, only: cross_val_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sequential_feature_selector
  type :: sequential_feature_selector
    integer :: n_features_to_select=0,cv=5,n_features_in_=0,n_features_to_select_=0
    character(8) :: direction='forward'
    logical :: use_tol=.false.
    real(dp) :: tol=0
    logical, allocatable :: support_(:)
  contains
    procedure :: fit=>sequential_fit
    procedure :: transform=>sequential_transform
    procedure :: inverse_transform=>sequential_inverse
    procedure :: get_support=>sequential_support
    procedure :: fit_transform=>sequential_fit_transform
  end type
contains
  subroutine sequential_fit(self,x,y,estimator)
    class(sequential_feature_selector), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(linear_regression), intent(in) :: estimator
    logical, allocatable :: current(:),candidate(:)
    integer, allocatable :: ids(:)
    real(dp), allocatable :: scores(:)
    real(dp) :: best_score,score,old_score
    integer :: p,target,iterations,step,j,best
    logical :: backward,auto_stop
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    p=size(x,2)
    call require(p>=2,'SequentialFeatureSelector requires at least two features')
    call require(self%cv>=2.and.self%cv<=size(y)/2,'SequentialFeatureSelector CV needs at least two test rows per fold')
    call require(self%n_features_to_select>=0.and.self%n_features_to_select<p, &
      'SequentialFeatureSelector feature count must be zero (auto) or between 1 and p-1')
    call require(self%direction=='forward'.or.self%direction=='backward','unknown sequential selection direction')
    call require(ieee_is_finite(self%tol),'nonfinite sequential selection tolerance')
    backward=self%direction=='backward'
    if(self%use_tol.and..not.backward) call require(self%tol>=0,'forward sequential tolerance must be nonnegative')
    target=self%n_features_to_select; auto_stop=target==0.and.self%use_tol
    if(target==0) then
      target=p/2
      if(auto_stop) target=p-1
    end if
    iterations=target
    ! sklearn auto backward selection performs target removals, including floor(p/2).
    if(backward.and.self%n_features_to_select/=0) iterations=p-target
    allocate(current(p),candidate(p)); current=.false.; ids=[(j,j=1,p)]
    old_score=-huge(1._dp)
    do step=1,iterations
      best=0; best_score=-huge(1._dp)
      do j=1,p
        if(current(j)) cycle
        candidate=current; candidate(j)=.true.
        if(backward) candidate=.not.candidate
        scores=cross_val_score(estimator,x(:,pack(ids,candidate)),y,cv=self%cv)
        call require(all(ieee_is_finite(scores)),'nonfinite sequential cross-validation score')
        score=sum(scores)/size(scores)
        if(best==0.or.score>best_score) then
          best=j; best_score=score
        end if
      end do
      if(auto_stop.and.step>1) then
        if(best_score-old_score<self%tol) exit
      end if
      current(best)=.true.; old_score=best_score
    end do
    if(backward) current=.not.current
    self%support_=current; self%n_features_in_=p; self%n_features_to_select_=count(current)
  end subroutine
  function sequential_transform(self,x) result(z)
    class(sequential_feature_selector), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: j
    call require(allocated(self%support_),'SequentialFeatureSelector is not fitted')
    call check_x(x)
    call require(size(x,2)==size(self%support_),'SequentialFeatureSelector feature count mismatch')
    z=x(:,pack([(j,j=1,size(x,2))],self%support_))
  end function
  function sequential_inverse(self,x) result(z)
    class(sequential_feature_selector), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: j
    call require(allocated(self%support_),'SequentialFeatureSelector is not fitted')
    call check_x(x)
    call require(size(x,2)==count(self%support_),'SequentialFeatureSelector inverse feature count mismatch')
    allocate(z(size(x,1),size(self%support_))); z=0
    z(:,pack([(j,j=1,size(self%support_))],self%support_))=x
  end function
  function sequential_support(self) result(mask)
    class(sequential_feature_selector), intent(in) :: self
    logical, allocatable :: mask(:)
    call require(allocated(self%support_),'SequentialFeatureSelector is not fitted'); mask=self%support_
  end function
  function sequential_fit_transform(self,x,y,estimator) result(z)
    class(sequential_feature_selector), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(linear_regression), intent(in) :: estimator
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y,estimator); z=self%transform(x)
  end function
end module
