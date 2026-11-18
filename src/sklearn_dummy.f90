module sklearn_dummy
  use sklearn_common
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: dummy_classifier,dummy_regressor
  type :: dummy_classifier
    character(16) :: strategy='prior'
    integer :: constant=0,random_state=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: class_prior_(:)
  contains
    procedure :: fit=>dummy_class_fit
    procedure :: predict_proba=>dummy_class_proba
    procedure :: predict=>dummy_class_predict
    procedure :: score=>dummy_class_score
  end type
  type :: dummy_regressor
    character(16) :: strategy='mean'
    real(dp) :: constant=0,quantile=0.5_dp,constant_=0
    logical :: fitted_=.false.
  contains
    procedure :: fit=>dummy_reg_fit
    procedure :: predict=>dummy_reg_predict
    procedure :: score=>dummy_reg_score
  end type
contains
  subroutine dummy_class_fit(self,x,y)
    class(dummy_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer :: c
    call check_xy(x,size(y)); self%classes_=unique_labels(y)
    self%class_prior_=spread(0._dp,1,size(self%classes_))
    do c=1,size(self%classes_)
      self%class_prior_(c)=real(count(y==self%classes_(c)),dp)/size(y)
    end do
    call require(any(self%strategy==['prior        ','most_frequent','stratified   ', &
      'uniform      ','constant     ']),'unknown dummy classifier strategy')
    if(self%strategy=='constant') call require(label_index(self%classes_,self%constant)>0,'unknown dummy constant')
  end subroutine
  function dummy_class_proba(self,x) result(prob)
    class(dummy_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    real(dp) :: r,cumulative
    integer :: i,c,k
    call require(allocated(self%classes_),'dummy classifier is not fitted')
    k=size(self%classes_); allocate(prob(size(x,1),k)); prob=0
    select case(trim(self%strategy))
    case('prior'); prob=spread(self%class_prior_,1,size(x,1))
    case('uniform'); prob=1._dp/k
    case('most_frequent'); prob(:,maxloc(self%class_prior_,dim=1))=1
    case('constant'); prob(:,label_index(self%classes_,self%constant))=1
    case('stratified')
      call seed_rng(self%random_state)
      do i=1,size(x,1)
        call random_number(r); cumulative=0
        do c=1,k
          cumulative=cumulative+self%class_prior_(c)
          if(r<cumulative.or.c==k) then
            prob(i,c)=1; exit
          end if
        end do
      end do
    end select
  end function
  function dummy_class_predict(self,x) result(y)
    class(dummy_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i,k
    real(dp), allocatable :: prob(:,:)
    real(dp) :: r
    prob=self%predict_proba(x); k=size(prob,2)
    if(self%strategy=='uniform') call seed_rng(self%random_state)
    do i=1,size(y)
      if(self%strategy=='uniform') then
        call random_number(r); y(i)=self%classes_(min(k,1+int(r*k)))
      else
        y(i)=self%classes_(maxloc(prob(i,:),dim=1))
      end if
    end do
  end function
  real(dp) function dummy_class_score(self,x,y) result(v)
    class(dummy_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine dummy_reg_fit(self,x,y)
    class(dummy_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer :: order(size(y)),lo,hi
    real(dp) :: position,q
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    select case(trim(self%strategy))
    case('mean'); self%constant_=sum(y)/size(y)
    case('constant'); self%constant_=self%constant
    case('median','quantile')
      q=self%quantile; if(self%strategy=='median') q=0.5_dp
      call require(q>=0.and.q<=1,'dummy quantile must be between 0 and 1')
      order=argsort(y); position=1+(size(y)-1)*q; lo=floor(position); hi=ceiling(position)
      self%constant_=y(order(lo))+(position-lo)*(y(order(hi))-y(order(lo)))
    case default; call require(.false.,'unknown dummy regression strategy')
    end select
    self%fitted_=.true.
  end subroutine
  function dummy_reg_predict(self,x) result(y)
    class(dummy_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(self%fitted_,'dummy regressor is not fitted'); y=self%constant_
  end function
  real(dp) function dummy_reg_score(self,x,y) result(v)
    class(dummy_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
