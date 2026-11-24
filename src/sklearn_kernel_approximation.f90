! Dense kernel feature maps following sklearn/kernel_approximation.py.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_kernel_approximation
  use sklearn_polynomial_count_sketch, only: polynomial_count_sketch
  use sklearn_sparse, only: csr_matrix
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use sklearn_common
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,cosine_similarity
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_fma
  implicit none
  private
  public :: polynomial_count_sketch
  public :: rbf_sampler,nystroem,additive_chi2_sampler,skewed_chi2_sampler
  ! Chi2 map contracts follow sklearn 1.6.1 kernel_approximation.py.
  type :: additive_chi2_sampler
    integer :: sample_steps=2,n_features_in_=0
    real(dp) :: sample_interval=-1
  contains
    procedure :: fit=>additive_fit
    procedure :: transform=>additive_transform
    procedure :: fit_transform=>additive_fit_transform
  end type
  type :: skewed_chi2_sampler
    integer :: n_components=100,random_state=0,n_features_in_=0
    real(dp) :: skewedness=1
    real(dp), allocatable :: random_weights_(:,:),random_offset_(:)
  contains
    procedure :: fit=>skewed_fit
    procedure :: transform=>skewed_transform
    procedure :: fit_transform=>skewed_fit_transform
  end type
  type :: rbf_sampler
    integer :: n_components=100,random_state=0,n_features_in_=0
    real(dp) :: gamma=1,gamma_=1
    logical :: gamma_scale=.false.
    real(dp), allocatable :: random_weights_(:,:),random_offset_(:)
  contains
    procedure, private :: fit_dense=>rbfs_fit
    procedure, private :: fit_csr=>rbfs_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure, private :: transform_dense=>rbfs_transform
    procedure, private :: transform_csr=>rbfs_transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure, private :: fit_transform_dense=>rbfs_fit_transform
    procedure, private :: fit_transform_csr=>rbfs_fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
  end type
  type :: nystroem
    integer :: n_components=100,random_state=0,degree=3
    character(16) :: kernel='rbf'
    real(dp) :: gamma=-1,coef0=1,gamma_=0
    real(dp), allocatable :: components_(:,:),normalization_(:,:)
    integer, allocatable :: component_indices_(:)
    character(16), private :: kernel_fit_='rbf'
    integer, private :: degree_fit_=3
    real(dp), private :: coef0_fit_=1
  contains
    procedure :: fit=>ny_fit
    procedure :: transform=>ny_transform
    procedure :: fit_transform=>ny_fit_transform
  end type
contains
  recursive real(dp) function rbf_numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8),left,right
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=ieee_difference(partial(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
      right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(rbf_numpy_sum(a(:split)),-rbf_numpy_sum(a(split+1:)))
    end if
  end function
  subroutine rbf_controls(self)
    class(rbf_sampler),intent(in)::self
    call require(self%n_components>0,'RBF sampler n_components must be positive')
    if(.not.self%gamma_scale) then
      call require(ieee_is_finite(self%gamma),'RBF sampler gamma must be finite')
      call require(self%gamma>=0,'invalid RBF sampler gamma')
    end if
    call require(self%random_state>=0,'RBF sampler random_state must be nonnegative')
  end subroutine
  subroutine rbf_csr_input(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'RBF sampler CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'RBF sampler CSR input contains NaN or infinity')
  end subroutine
  subroutine rbf_generate(self,p,variance)
    class(rbf_sampler),intent(inout)::self
    integer,intent(in)::p
    real(dp),intent(in)::variance
    type(legacy_random_state)::rng
    real(dp)::scale,square
    integer::i,j
    self%n_features_in_=p;self%gamma_=self%gamma
    if(self%gamma_scale) then
      self%gamma_=1
      if(ieee_is_nan(variance)) then
        self%gamma_=variance
      else if(variance/=0) then
        self%gamma_=ieee_quotient(1._dp,ieee_product(real(p,dp),variance))
      end if
    end if
    square=ieee_product(2._dp,self%gamma_);scale=ieee_value(0._dp,ieee_quiet_nan)
    if(.not.ieee_is_nan(square)) then
      if(square>=0) scale=sqrt(square)
    end if
    if(allocated(self%random_weights_)) deallocate(self%random_weights_)
    if(allocated(self%random_offset_)) deallocate(self%random_offset_)
    allocate(self%random_weights_(p,self%n_components),self%random_offset_(self%n_components))
    call rng%seed(self%random_state)
    do i=1,p
      do j=1,self%n_components
        self%random_weights_(i,j)=ieee_product(scale,rng%normal())
      end do
    end do
    do j=1,self%n_components
      self%random_offset_(j)=rng%uniform()*(2*acos(-1._dp))
    end do
  end subroutine
  subroutine rbfs_fit(self,x)
    class(rbf_sampler),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::values(:)
    real(dp)::variance,mean,total
    integer::i
    call check_x(x);call rbf_controls(self);variance=0
    if(self%gamma_scale) then
      values=reshape(transpose(x),[size(x)]);total=real(size(values),dp)
      mean=ieee_quotient(rbf_numpy_sum(values),total)
      do i=1,size(values)
        values(i)=ieee_difference(values(i),mean)
        values(i)=ieee_product(values(i),values(i))
      end do
      variance=ieee_quotient(rbf_numpy_sum(values),total)
    end if
    call rbf_generate(self,size(x,2),variance)
  end subroutine
  subroutine rbfs_fit_csr(self,x)
    class(rbf_sampler),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::row_values(:),means(:),squares(:)
    integer,allocatable::seen(:),columns(:)
    real(dp)::factor,mean,variance,value
    integer::i,k,j,n,previous,index_,first,last,step
    logical::canonical
    call rbf_csr_input(x);call rbf_controls(self);variance=0
    if(self%gamma_scale) then
      factor=1/(real(x%n_rows,dp)*real(x%n_cols,dp))
      allocate(row_values(x%n_cols),seen(x%n_cols),columns(x%n_cols),means(x%n_rows),squares(x%n_rows))
      seen=0;canonical=.true.
      do i=1,x%n_rows
        previous=0
        do k=x%indptr(i),x%indptr(i+1)-1
          if(x%indices(k)<=previous) canonical=.false.
          previous=x%indices(k)
        end do
      end do
      do i=1,x%n_rows
        n=0;means(i)=0;squares(i)=0
        do k=x%indptr(i),x%indptr(i+1)-1
          j=x%indices(k)
          if(seen(j)/=i) then
            n=n+1;columns(n)=j;seen(j)=i;row_values(j)=0
          end if
          row_values(j)=ieee_difference(row_values(j),-x%data(k))
          means(i)=ieee_difference(means(i),-ieee_product(x%data(k),factor))
        end do
        first=1;last=n;step=1
        if(.not.canonical) then
          first=n;last=1;step=-1
        end if
        do index_=first,last,step
          value=row_values(columns(index_));value=ieee_product(value,value)
          squares(i)=ieee_difference(squares(i),-ieee_product(value,factor))
        end do
      end do
      mean=rbf_numpy_sum(means)
      variance=ieee_difference(rbf_numpy_sum(squares),ieee_product(mean,mean))
    end if
    call rbf_generate(self,x%n_cols,variance)
  end subroutine
  subroutine rbf_finish(self,z)
    class(rbf_sampler),intent(in)::self
    real(dp),intent(inout)::z(:,:)
    real(dp)::phase,scale
    integer::i,j
    call require(self%n_components>0,'RBF sampler transform n_components must be positive')
    scale=sqrt(2._dp/self%n_components)
    do j=1,size(z,2)
      do i=1,size(z,1)
        phase=ieee_difference(z(i,j),-self%random_offset_(j))
        z(i,j)=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(phase)) z(i,j)=cos(phase)*scale
      end do
    end do
  end subroutine
  function rbfs_transform(self,x) result(z)
    class(rbf_sampler),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    real(dp)::bound
    integer::i,j,k
    logical::fast,fused
    call require(allocated(self%random_weights_),'RBF sampler is not fitted')
    call check_xy(transpose(x),size(self%random_weights_,1))
    fast=.false.;fused=.false.
    if(all(ieee_is_finite(self%random_weights_))) then
      bound=ieee_product(ieee_product(maxval(abs(x)),maxval(abs(self%random_weights_))),real(size(x,2),dp))
      ! Large phases amplify last-bit reassociation into different cosine values.
      ! Keep ordinary maps on MATMUL and control summation for sensitive phases.
      fast=.false.
      if(ieee_is_finite(bound)) then
        fast=bound<=1024._dp;fused=.not.fast
      end if
    end if
    if(fast) then
      z=matmul(x,self%random_weights_)
    else
      allocate(z(size(x,1),size(self%random_weights_,2)));z=0
      do j=1,size(z,2)
        do i=1,size(z,1)
          do k=1,size(x,2)
            if(fused) then
              z(i,j)=ieee_fma(x(i,k),self%random_weights_(k,j),z(i,j))
            else
              z(i,j)=ieee_difference(z(i,j),-ieee_product(x(i,k),self%random_weights_(k,j)))
            end if
          end do
        end do
      end do
    end if
    call rbf_finish(self,z)
  end function
  function rbfs_transform_csr(self,x) result(z)
    class(rbf_sampler),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call require(allocated(self%random_weights_),'RBF sampler is not fitted')
    call rbf_csr_input(x)
    call require(x%n_cols==size(self%random_weights_,1),'RBF sampler CSR feature count mismatch')
    z=x%matmul(self%random_weights_);call rbf_finish(self,z)
  end function
  function rbfs_fit_transform(self,x) result(z)
    class(rbf_sampler),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  function rbfs_fit_transform_csr(self,x) result(z)
    class(rbf_sampler),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  function kernel_matrix(self,x,y) result(k)
    class(nystroem), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(self%kernel_fit_)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('linear'); k=linear_kernel(x,y)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree_fit_,self%gamma_,self%coef0_fit_)
    case('cosine'); k=cosine_similarity(x,y)
    end select
  end function
  subroutine ny_fit(self,x)
    class(nystroem), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: gram(:,:),values(:),vectors(:,:)
    integer :: n,k,i,j,tmp
    integer, allocatable :: order(:)
    real(dp) :: draw
    call check_x(x); n=size(x,1)
    call require(self%n_components>0,'Nystroem n_components must be positive')
    call require(self%kernel=='rbf'.or.self%kernel=='linear'.or.self%kernel=='poly'.or. &
      self%kernel=='polynomial'.or.self%kernel=='cosine','unsupported Nystroem kernel')
    call require(ieee_is_finite(self%gamma).and.(self%gamma>=0.or.self%gamma==-1), &
      'invalid Nystroem gamma')
    call require(self%degree>=0.and.ieee_is_finite(self%coef0).and.self%coef0>=0, &
      'invalid Nystroem polynomial parameters')
    self%kernel_fit_=self%kernel; self%degree_fit_=self%degree; self%coef0_fit_=self%coef0
    self%gamma_=self%gamma
    if(self%gamma_==-1) self%gamma_=1._dp/size(x,2)
    k=min(n,self%n_components); allocate(order(n)); order=[(i,i=1,n)]
    call seed_rng(self%random_state)
    do i=n,2,-1
      call random_number(draw); j=1+int(draw*i); tmp=order(i); order(i)=order(j); order(j)=tmp
    end do
    self%component_indices_=order(:k); self%components_=x(order(:k),:)
    gram=kernel_matrix(self,self%components_,self%components_)
    call require(all(ieee_is_finite(gram)),'Nystroem nonfinite kernel matrix')
    allocate(values(k),vectors(k,k)); call symmetric_eigen(gram,values,vectors)
    call require(minval(values)>=-1.e-10_dp*max(1._dp,maxval(abs(values))), &
      'Nystroem basis kernel must be positive semidefinite')
    values=1/sqrt(max(values,1.e-12_dp))
    self%normalization_=matmul(vectors*spread(values,1,k),transpose(vectors))
  end subroutine
  function ny_transform(self,x) result(z)
    class(nystroem), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'Nystroem is not fitted')
    call check_xy(transpose(x),size(self%components_,2))
    z=matmul(kernel_matrix(self,x,self%components_),transpose(self%normalization_))
  end function
  function ny_fit_transform(self,x) result(z)
    class(nystroem), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  real(dp) function additive_interval(self) result(h)
    class(additive_chi2_sampler), intent(in) :: self
    call require(self%sample_steps>=1,'AdditiveChi2 sample_steps must be positive')
    call require(ieee_is_finite(self%sample_interval),'nonfinite AdditiveChi2 sample_interval')
    call require(self%sample_interval==-1.or.self%sample_interval>=0,'invalid AdditiveChi2 sample_interval')
    h=self%sample_interval
    if(h==-1) then
      select case(self%sample_steps)
      case(1); h=.8_dp
      case(2); h=.5_dp
      case(3); h=.4_dp
      case default; call require(.false.,'AdditiveChi2 needs an explicit interval for these steps')
      end select
    end if
  end function

  subroutine additive_fit(self,x)
    class(additive_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: interval
    interval=additive_interval(self)
    call check_x(x); call require(all(x>=0),'AdditiveChi2 requires nonnegative input')
    self%n_features_in_=size(x,2)
  end subroutine

  function additive_transform(self,x) result(z)
    use, intrinsic :: iso_fortran_env, only: int64
    class(additive_chi2_sampler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    real(dp) :: h,a,amplitude,phase,pi,log_amplitude
    integer :: n,p,i,j,k,offset,width
    integer(int64) :: wide
    h=additive_interval(self); call check_x(x)
    call require(all(x>=0),'AdditiveChi2 requires nonnegative input')
    n=size(x,1); p=size(x,2)
    if(self%n_features_in_>0) call require(p==self%n_features_in_,'AdditiveChi2 feature count mismatch')
    wide=(2_int64*self%sample_steps-1)*int(p,int64)
    call require(wide<=huge(1),'AdditiveChi2 output width overflow')
    width=int(wide); allocate(z(n,width)); z=0
    if(h==0) return
    pi=acos(-1._dp)
    do j=1,p
      do i=1,n
        if(x(i,j)==0) cycle
        z(i,j)=sqrt(x(i,j))*sqrt(h)
        do k=1,self%sample_steps-1
          ! Beyond this limit all harmonics underflow for any finite input.
          if(h>3000._dp/(pi*real(k,dp))) cycle
          a=pi*real(k,dp)*h
          log_amplitude=.5_dp*(log(4._dp)+log(x(i,j))+log(h)-a-log(1+exp(-2*a)))
          amplitude=exp(log_amplitude); phase=real(k,dp)*h*log(x(i,j))
          offset=(2*k-1)*p
          z(i,offset+j)=amplitude*cos(phase)
          z(i,offset+p+j)=amplitude*sin(phase)
        end do
      end do
    end do
  end function

  function additive_fit_transform(self,x) result(z)
    class(additive_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function

  subroutine skewed_fit(self,x)
    class(skewed_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: pi
    call check_x(x)
    call require(self%n_components>=1,'SkewedChi2 n_components must be positive')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    self%n_features_in_=size(x,2); pi=acos(-1._dp)
    if(allocated(self%random_weights_)) deallocate(self%random_weights_,self%random_offset_)
    allocate(self%random_weights_(size(x,2),self%n_components),self%random_offset_(self%n_components))
    call seed_rng(self%random_state); call random_number(self%random_weights_)
    ! Exclude the possible zero endpoint before inverse-CDF sampling.
    self%random_weights_=max(tiny(1._dp),self%random_weights_)
    self%random_weights_=log(tan(.5_dp*pi*self%random_weights_))/pi
    call random_number(self%random_offset_); self%random_offset_=self%random_offset_*2*pi
  end subroutine

  function skewed_transform(self,x) result(z)
    class(skewed_chi2_sampler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),shifted(:,:)
    integer :: i,j
    call require(allocated(self%random_weights_),'SkewedChi2 is not fitted')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    call check_xy(transpose(x),self%n_features_in_)
    call require(all(x>-self%skewedness),'SkewedChi2 input must exceed -skewedness')
    allocate(shifted(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(x(i,j)>0.and.self%skewedness>0) then
          shifted(i,j)=log(max(x(i,j),self%skewedness))+ &
            log(1+min(x(i,j),self%skewedness)/max(x(i,j),self%skewedness))
        else
          shifted(i,j)=log(x(i,j)+self%skewedness)
        end if
      end do
    end do
    z=cos(matmul(shifted,self%random_weights_)+spread(self%random_offset_,1,size(x,1)))* &
      (sqrt(2._dp)/sqrt(real(size(self%random_offset_),dp)))
  end function

  function skewed_fit_transform(self,x) result(z)
    class(skewed_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
