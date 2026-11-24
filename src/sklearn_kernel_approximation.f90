! Dense kernel feature maps following sklearn/kernel_approximation.py.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_kernel_approximation
  use sklearn_nystroem_svd,only:kernel_svd
  use sklearn_polynomial_count_sketch, only: polynomial_count_sketch
  use sklearn_sparse, only: csr_matrix
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use sklearn_common
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_fma,ieee_negative_inf
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
  abstract interface
    function nystroem_kernel(a,b) result(value)
      import dp
      real(dp),intent(in)::a(:),b(:)
      real(dp)::value
    end function
  end interface
  public::nystroem_kernel
  type :: nystroem
    integer :: n_components=100,random_state=0,degree=3,n_features_in_=0
    character(16) :: kernel='rbf'
    real(dp) :: gamma=-1,coef0=1,gamma_=0
    real(dp), allocatable :: components_(:,:),normalization_(:,:)
    integer, allocatable :: component_indices_(:)
    procedure(nystroem_kernel),pointer,nopass::kernel_callback=>null()
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
  function fourier_dot(x,weights) result(z)
    real(dp),intent(in)::x(:,:),weights(:,:)
    real(dp),allocatable::z(:,:)
    real(dp)::bound
    integer::i,j,k
    logical::fast,fused
    fast=.false.;fused=.false.
    if(all(ieee_is_finite(weights)).and.all(ieee_is_finite(x))) then
      bound=ieee_product(ieee_product(maxval(abs(x)),maxval(abs(weights))),real(size(x,2),dp))
      ! Large phases amplify last-bit reassociation into different cosine values.
      ! Keep ordinary maps on MATMUL and control summation for sensitive phases.
      fast=.false.
      if(ieee_is_finite(bound)) then
        fast=bound<=1024._dp;fused=.not.fast
      end if
    end if
    if(fast) then
      z=matmul(x,weights)
    else
      allocate(z(size(x,1),size(weights,2)));z=0
      do j=1,size(z,2)
        do i=1,size(z,1)
          do k=1,size(x,2)
            if(fused) then
              z(i,j)=ieee_fma(x(i,k),weights(k,j),z(i,j))
            else
              z(i,j)=ieee_difference(z(i,j),-ieee_product(x(i,k),weights(k,j)))
            end if
          end do
        end do
      end do
    end if
  end function
  function rbfs_transform(self,x) result(z)
    class(rbf_sampler),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call require(allocated(self%random_weights_),'RBF sampler is not fitted')
    call check_xy(transpose(x),size(self%random_weights_,1))
    z=fourier_dot(x,self%random_weights_)
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
  subroutine ny_controls(self)
    class(nystroem),intent(in)::self
    call require(any(self%kernel==[character(16)::'rbf','linear','poly','polynomial','cosine', &
      'laplacian','sigmoid','chi2','additive_chi2','precomputed','callable']),'unsupported Nystroem kernel')
    call require(ieee_is_finite(self%gamma),'invalid Nystroem gamma')
    call require(self%gamma>=0.or.self%gamma==-1,'invalid Nystroem gamma')
    call require(self%degree>=1.and.ieee_is_finite(self%coef0),'invalid Nystroem polynomial parameters')
    if(self%kernel=='callable'.or.self%kernel=='precomputed') then
      call require(self%gamma==-1.and.self%degree==3.and.self%coef0==1, &
        'Nystroem callable/precomputed kernels require default gamma, degree and coef0')
    end if
    if(self%kernel=='callable') call require(associated(self%kernel_callback),'Nystroem kernel callback is not associated')
  end subroutine
  function ny_cosine_input(x) result(z)
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    real(dp)::norm
    integer::i,j
    allocate(z(size(x,1),size(x,2)))
    do i=1,size(x,1)
      norm=0
      do j=1,size(x,2)
        norm=ieee_difference(norm,-ieee_product(x(i,j),x(i,j)))
      end do
      norm=sqrt(norm)
      if(norm<10*epsilon(1._dp)) norm=1
      do j=1,size(x,2)
        z(i,j)=ieee_quotient(x(i,j),norm)
      end do
    end do
  end function
  function kernel_matrix(self,x,y,symmetric) result(k)
    class(nystroem),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    logical,intent(in),optional::symmetric
    real(dp),allocatable::k(:,:),nx(:),ny(:)
    real(dp)::g,value,base,denom,difference
    integer::i,j,t
    logical::sym
    call ny_controls(self)
    g=self%gamma
    if(g==-1) then
      g=1._dp/size(x,2)
      if(self%kernel=='chi2') g=1
    end if
    sym=.false.;if(present(symmetric)) sym=symmetric
    select case(self%kernel)
    case('rbf')
      k=fourier_dot(x,transpose(y));allocate(nx(size(x,1)),ny(size(y,1)));nx=0;ny=0
      do i=1,size(x,1)
        do t=1,size(x,2)
          nx(i)=ieee_difference(nx(i),-ieee_product(x(i,t),x(i,t)))
        end do
      end do
      do j=1,size(y,1)
        do t=1,size(y,2)
          ny(j)=ieee_difference(ny(j),-ieee_product(y(j,t),y(j,t)))
        end do
      end do
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=ieee_product(-2._dp,k(i,j));value=ieee_difference(value,-nx(i))
          value=ieee_difference(value,-ny(j))
          if(.not.ieee_is_nan(value)) value=max(value,0._dp)
          if(sym.and.i==j) value=0
          value=ieee_product(-g,value);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
        end do
      end do
    case('linear');k=fourier_dot(x,transpose(y))
    case('poly','polynomial')
      k=fourier_dot(x,transpose(y))
      do j=1,size(y,1)
        do i=1,size(x,1)
          base=ieee_difference(ieee_product(g,k(i,j)),-self%coef0);value=1
          do t=1,self%degree
            value=ieee_product(value,base)
          end do
          k(i,j)=value
        end do
      end do
    case('cosine');k=fourier_dot(ny_cosine_input(x),transpose(ny_cosine_input(y)))
    case('laplacian')
      allocate(k(size(x,1),size(y,1)))
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=0
          do t=1,size(x,2)
            value=ieee_difference(value,-abs(ieee_difference(x(i,t),y(j,t))))
          end do
          value=ieee_product(-g,value);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
        end do
      end do
    case('sigmoid')
      k=fourier_dot(x,transpose(y))
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=ieee_difference(ieee_product(g,k(i,j)),-self%coef0);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=tanh(value)
        end do
      end do
    case('chi2','additive_chi2')
      call require(all(x>=0).and.all(y>=0),'Nystroem chi-squared kernels require nonnegative inputs')
      allocate(k(size(x,1),size(y,1)))
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=0
          do t=1,size(x,2)
            denom=ieee_difference(x(i,t),-y(j,t))
            if(denom==0) cycle
            difference=ieee_difference(x(i,t),y(j,t))
            base=ieee_quotient(ieee_product(difference,difference),denom)
            value=ieee_difference(value,-base)
          end do
          k(i,j)=-value
          if(self%kernel=='chi2') then
            value=ieee_product(g,k(i,j));k(i,j)=value
            if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
          end if
        end do
      end do
    case('precomputed')
      call require(size(x,2)==size(y,1),'Nystroem precomputed kernel shape mismatch')
      k=x
    case('callable')
      allocate(k(size(x,1),size(y,1)))
      do i=1,size(x,1)
        do j=1,size(y,1)
          if(sym.and.i>j) cycle
          k(i,j)=self%kernel_callback(x(i,:),y(j,:))
          if(sym) k(j,i)=k(i,j)
        end do
      end do
    end select
  end function
  subroutine ny_fit(self,x)
    class(nystroem),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(legacy_random_state)::rng
    real(dp),allocatable::gram(:,:),values(:),u(:,:),v(:,:)
    integer,allocatable::order(:)
    integer::n,k,i,j,tmp,status
    call check_x(x);n=size(x,1)
    call require(self%n_components>0,'Nystroem n_components must be positive')
    call require(self%random_state>=0,'Nystroem random_state must be nonnegative')
    call ny_controls(self);self%n_features_in_=size(x,2)
    self%gamma_=self%gamma
    if(self%gamma_==-1) then
      self%gamma_=1._dp/size(x,2)
      if(self%kernel=='chi2') self%gamma_=1
    end if
    k=min(n,self%n_components);order=[(i,i=1,n)];call rng%seed(self%random_state)
    do i=n,2,-1
      j=rng%integer_index(i);tmp=order(i);order(i)=order(j);order(j)=tmp
    end do
    self%component_indices_=order(:k);self%components_=x(order(:k),:)
    gram=kernel_matrix(self,self%components_,self%components_,.true.)
    call require(all(ieee_is_finite(gram)),'Nystroem nonfinite kernel matrix')
    allocate(values(k),u(k,k),v(k,k));call kernel_svd(gram,u,values,v,status,240)
    call require(status==0,'Nystroem Jacobi SVD did not converge')
    values=1/sqrt(max(values,1.e-12_dp))
    self%normalization_=matmul(u*spread(values,1,k),transpose(v))
  end subroutine
  function ny_transform(self,x) result(z)
    class(nystroem),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call require(allocated(self%components_),'Nystroem is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    z=fourier_dot(kernel_matrix(self,x,self%components_),transpose(self%normalization_))
  end function
  function ny_fit_transform(self,x) result(z)
    class(nystroem),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
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

  real(dp) function skewed_log(value) result(result_)
    real(dp),intent(in)::value
    result_=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(value)) return
    if(value==0) then
      result_=ieee_value(0._dp,ieee_negative_inf)
    else if(value>0) then
      result_=log(value)
    end if
  end function
  subroutine skewed_fit(self,x)
    class(skewed_chi2_sampler),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(legacy_random_state)::rng
    real(dp)::pi,u
    integer::i,j
    call check_x(x)
    call require(self%n_components>=1,'SkewedChi2 n_components must be positive')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    call require(self%random_state>=0,'SkewedChi2 random_state must be nonnegative')
    self%n_features_in_=size(x,2);pi=acos(-1._dp)
    if(allocated(self%random_weights_)) deallocate(self%random_weights_)
    if(allocated(self%random_offset_)) deallocate(self%random_offset_)
    allocate(self%random_weights_(size(x,2),self%n_components),self%random_offset_(self%n_components))
    call rng%seed(self%random_state)
    do i=1,size(x,2)
      do j=1,self%n_components
        u=rng%uniform()
        self%random_weights_(i,j)=(1/pi)*skewed_log(tan((pi/2)*u))
      end do
    end do
    do j=1,self%n_components
      self%random_offset_(j)=rng%uniform()*(2*pi)
    end do
  end subroutine
  function skewed_transform(self,x) result(z)
    class(skewed_chi2_sampler),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:),shifted(:,:)
    real(dp)::phase,scale
    integer::i,j
    call require(allocated(self%random_weights_),'SkewedChi2 is not fitted')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    call check_xy(transpose(x),self%n_features_in_)
    call require(all(x>-self%skewedness),'SkewedChi2 input must exceed -skewedness')
    call require(self%n_components>0,'SkewedChi2 transform n_components must be positive')
    allocate(shifted(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        shifted(i,j)=skewed_log(ieee_difference(x(i,j),-self%skewedness))
      end do
    end do
    z=fourier_dot(shifted,self%random_weights_)
    scale=sqrt(2._dp)/sqrt(real(self%n_components,dp))
    do j=1,size(z,2)
      do i=1,size(z,1)
        phase=ieee_difference(z(i,j),-self%random_offset_(j))
        z(i,j)=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(phase)) z(i,j)=cos(phase)*scale
      end do
    end do
  end function

  function skewed_fit_transform(self,x) result(z)
    class(skewed_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
