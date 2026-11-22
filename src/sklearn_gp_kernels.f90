! Covariance contracts from sklearn 1.6.1 gaussian_process/kernels.py (BSD-3-Clause).
! Flat expression storage and native interfaces are original MIT contributions.
module sklearn_gp_kernels
  use sklearn_common, only: dp,require,check_x
  use sklearn_bessel_k, only: log_besselk
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: gp_kernel,rbf_covariance,matern_covariance,rational_quadratic_covariance
  public :: periodic_covariance,dot_product_covariance,constant_covariance,white_covariance
  public :: operator(+),operator(*),operator(**)
  type :: kernel_node
    integer :: kind=1,left=0,right=0
    real(dp) :: first=1,nu=1.5_dp,exponent=1
    real(dp), allocatable :: lengths(:)
    logical :: fixed(2)=.false.
  end type
  type :: gp_kernel
    private
    type(kernel_node), allocatable :: nodes(:)
  contains
    procedure :: matrix=>kernel_matrix
    procedure :: diag=>kernel_diag
    procedure :: gradient=>kernel_gradient
    procedure :: n_dims=>kernel_dimensions
    procedure :: theta=>kernel_theta
    procedure :: with_theta=>kernel_with_theta
    procedure :: is_set=>kernel_is_set
  end type
  type :: evaluation
    real(dp), allocatable :: value(:,:),gradient(:,:,:)
  end type
  interface rbf_covariance
    module procedure rbf_scalar,rbf_vector
  end interface
  interface matern_covariance
    module procedure matern_scalar,matern_vector
  end interface
  interface operator(+)
    module procedure kernel_sum,kernel_sum_scalar,scalar_sum_kernel
  end interface
  interface operator(*)
    module procedure kernel_product,kernel_product_scalar,scalar_product_kernel
  end interface
  interface operator(**)
    module procedure kernel_power,kernel_power_integer
  end interface
contains
  logical function kernel_is_set(self)
    class(gp_kernel), intent(in) :: self
    kernel_is_set=allocated(self%nodes)
  end function
  subroutine check_positive(value,name,zero)
    real(dp), intent(in) :: value
    character(*), intent(in) :: name
    logical, optional, intent(in) :: zero
    call require(ieee_is_finite(value),'GP kernel '//name//' must be finite')
    if(present(zero)) then
      if(zero) then
        call require(value>=0,'GP kernel '//name//' must be nonnegative');return
      end if
    end if
    call require(value>0,'GP kernel '//name//' must be positive')
  end subroutine
  function length_kernel(kind,lengths,fixed) result(kernel)
    integer, intent(in) :: kind
    real(dp), intent(in) :: lengths(:)
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    call require(size(lengths)>0,'GP kernel lengths must be nonempty')
    call require(all(ieee_is_finite(lengths)),'GP kernel lengths must be finite')
    call require(all(lengths>0),'GP kernel lengths must be positive')
    allocate(kernel%nodes(1));kernel%nodes(1)%kind=kind;kernel%nodes(1)%lengths=lengths
    if(present(fixed)) kernel%nodes(1)%fixed(1)=fixed
  end function
  function rbf_scalar(length_scale,fixed) result(kernel)
    real(dp), optional, intent(in) :: length_scale
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    real(dp) :: length
    length=1;if(present(length_scale)) length=length_scale
    kernel=length_kernel(1,[length],fixed)
  end function
  function rbf_vector(length_scale,fixed) result(kernel)
    real(dp), intent(in) :: length_scale(:)
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    kernel=length_kernel(1,length_scale,fixed)
  end function
  function matern_scalar(length_scale,nu,fixed) result(kernel)
    real(dp), optional, intent(in) :: length_scale,nu
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    real(dp) :: length
    length=1;if(present(length_scale)) length=length_scale
    kernel=length_kernel(2,[length],fixed)
    if(present(nu)) kernel%nodes(1)%nu=nu
    call require(.not.ieee_is_nan(kernel%nodes(1)%nu),'GP kernel nu must not be NaN')
    call require(kernel%nodes(1)%nu>0,'GP kernel nu must be positive')
  end function
  function matern_vector(length_scale,nu,fixed) result(kernel)
    real(dp), intent(in) :: length_scale(:)
    real(dp), optional, intent(in) :: nu
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    kernel=length_kernel(2,length_scale,fixed)
    if(present(nu)) kernel%nodes(1)%nu=nu
    call require(.not.ieee_is_nan(kernel%nodes(1)%nu),'GP kernel nu must not be NaN')
    call require(kernel%nodes(1)%nu>0,'GP kernel nu must be positive')
  end function
  function rational_quadratic_covariance(length_scale,alpha,fixed_length,fixed_alpha) result(kernel)
    real(dp), optional, intent(in) :: length_scale,alpha
    logical, optional, intent(in) :: fixed_length,fixed_alpha
    type(gp_kernel) :: kernel
    real(dp) :: length,a
    length=1;a=1
    if(present(length_scale)) length=length_scale
    if(present(alpha)) a=alpha
    kernel=length_kernel(3,[length]);kernel%nodes(1)%first=a
    call check_positive(a,'alpha')
    if(present(fixed_alpha)) kernel%nodes(1)%fixed(1)=fixed_alpha
    if(present(fixed_length)) kernel%nodes(1)%fixed(2)=fixed_length
  end function
  function periodic_covariance(length_scale,periodicity,fixed_length,fixed_periodicity) result(kernel)
    real(dp), optional, intent(in) :: length_scale,periodicity
    logical, optional, intent(in) :: fixed_length,fixed_periodicity
    type(gp_kernel) :: kernel
    real(dp) :: length,period
    length=1;period=1
    if(present(length_scale)) length=length_scale
    if(present(periodicity)) period=periodicity
    kernel=length_kernel(4,[length]);kernel%nodes(1)%first=period
    call check_positive(period,'periodicity')
    if(present(fixed_length)) kernel%nodes(1)%fixed(1)=fixed_length
    if(present(fixed_periodicity)) kernel%nodes(1)%fixed(2)=fixed_periodicity
  end function
  function simple_kernel(kind,value,fixed) result(kernel)
    integer, intent(in) :: kind
    real(dp), intent(in) :: value
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    call check_positive(value,'value',zero=.true.)
    allocate(kernel%nodes(1));kernel%nodes(1)%kind=kind;kernel%nodes(1)%first=value
    if(present(fixed)) kernel%nodes(1)%fixed(1)=fixed
  end function
  function dot_product_covariance(sigma_0,fixed) result(kernel)
    real(dp), optional, intent(in) :: sigma_0
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    real(dp) :: value
    value=1;if(present(sigma_0)) value=sigma_0
    kernel=simple_kernel(5,value,fixed)
  end function
  function constant_covariance(constant_value,fixed) result(kernel)
    real(dp), optional, intent(in) :: constant_value
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    real(dp) :: value
    value=1;if(present(constant_value)) value=constant_value
    kernel=simple_kernel(6,value,fixed)
  end function
  function white_covariance(noise_level,fixed) result(kernel)
    real(dp), optional, intent(in) :: noise_level
    logical, optional, intent(in) :: fixed
    type(gp_kernel) :: kernel
    real(dp) :: value
    value=1;if(present(noise_level)) value=noise_level
    kernel=simple_kernel(7,value,fixed)
  end function
  function combine(a,b,kind) result(kernel)
    type(gp_kernel), intent(in) :: a,b
    integer, intent(in) :: kind
    type(gp_kernel) :: kernel
    type(kernel_node) :: root
    integer :: i,n,m
    call require(a%is_set().and.b%is_set(),'GP kernel expression contains an empty child')
    n=size(a%nodes);m=size(b%nodes);root%kind=kind;root%left=n;root%right=n+m
    kernel%nodes=[a%nodes,b%nodes,root]
    do i=n+1,n+m
      if(kernel%nodes(i)%left>0) kernel%nodes(i)%left=kernel%nodes(i)%left+n
      if(kernel%nodes(i)%right>0) kernel%nodes(i)%right=kernel%nodes(i)%right+n
    end do
  end function
  function kernel_sum(a,b) result(kernel)
    type(gp_kernel), intent(in) :: a,b
    type(gp_kernel) :: kernel
    kernel=combine(a,b,8)
  end function
  function kernel_product(a,b) result(kernel)
    type(gp_kernel), intent(in) :: a,b
    type(gp_kernel) :: kernel
    kernel=combine(a,b,9)
  end function
  function kernel_sum_scalar(a,b) result(kernel)
    type(gp_kernel), intent(in) :: a
    real(dp), intent(in) :: b
    type(gp_kernel) :: kernel
    kernel=combine(a,constant_covariance(b),8)
  end function
  function scalar_sum_kernel(a,b) result(kernel)
    real(dp), intent(in) :: a
    type(gp_kernel), intent(in) :: b
    type(gp_kernel) :: kernel
    kernel=combine(constant_covariance(a),b,8)
  end function
  function kernel_product_scalar(a,b) result(kernel)
    type(gp_kernel), intent(in) :: a
    real(dp), intent(in) :: b
    type(gp_kernel) :: kernel
    kernel=combine(a,constant_covariance(b),9)
  end function
  function scalar_product_kernel(a,b) result(kernel)
    real(dp), intent(in) :: a
    type(gp_kernel), intent(in) :: b
    type(gp_kernel) :: kernel
    kernel=combine(constant_covariance(a),b,9)
  end function
  function kernel_power_integer(a,exponent) result(kernel)
    type(gp_kernel), intent(in) :: a
    integer, intent(in) :: exponent
    type(gp_kernel) :: kernel
    kernel=kernel_power(a,real(exponent,dp))
  end function
  function kernel_power(a,exponent) result(kernel)
    type(gp_kernel), intent(in) :: a
    real(dp), intent(in) :: exponent
    type(gp_kernel) :: kernel
    type(kernel_node) :: root
    call require(a%is_set(),'GP kernel expression contains an empty child')
    call require(ieee_is_finite(exponent),'GP kernel exponent must be finite')
    root%kind=10;root%left=size(a%nodes);root%exponent=exponent
    kernel%nodes=[a%nodes,root]
  end function
  integer function leaf_dimensions(node) result(n)
    type(kernel_node), intent(in) :: node
    n=0
    select case(node%kind)
    case(1,2)
      if(.not.node%fixed(1)) n=size(node%lengths)
    case(3,4);n=count(.not.node%fixed)
    case(5:7);n=merge(0,1,node%fixed(1))
    end select
  end function
  integer function kernel_dimensions(self) result(n)
    class(gp_kernel), intent(in) :: self
    integer :: i
    call require(self%is_set(),'GP kernel is empty');n=0
    do i=1,size(self%nodes)
      n=n+leaf_dimensions(self%nodes(i))
    end do
  end function
  function parameter_values(node) result(values)
    type(kernel_node), intent(in) :: node
    real(dp), allocatable :: values(:)
    allocate(values(0))
    select case(node%kind)
    case(1,2)
      if(.not.node%fixed(1)) values=node%lengths
    case(3)
      if(.not.node%fixed(1)) values=[values,node%first]
      if(.not.node%fixed(2)) values=[values,node%lengths(1)]
    case(4)
      if(.not.node%fixed(1)) values=[values,node%lengths(1)]
      if(.not.node%fixed(2)) values=[values,node%first]
    case(5:7)
      if(.not.node%fixed(1)) values=[node%first]
    end select
  end function
  function kernel_theta(self) result(theta)
    class(gp_kernel), intent(in) :: self
    real(dp), allocatable :: theta(:),values(:)
    integer :: i,j
    call require(self%is_set(),'GP kernel is empty');allocate(theta(0))
    do i=1,size(self%nodes)
      values=parameter_values(self%nodes(i))
      do j=1,size(values)
        if(values(j)==0) then
          values(j)=ieee_value(0._dp,ieee_negative_inf)
        else
          values(j)=log(values(j))
        end if
      end do
      theta=[theta,values]
    end do
  end function
  function kernel_with_theta(self,theta) result(kernel)
    class(gp_kernel), intent(in) :: self
    real(dp), intent(in) :: theta(:)
    type(gp_kernel) :: kernel
    real(dp), allocatable :: values(:)
    integer :: i,j,n,pos
    call require(size(theta)==self%n_dims(),'GP kernel theta length mismatch')
    call require(.not.any(ieee_is_nan(theta)),'GP kernel theta must not contain NaN')
    call require(all(theta<=log(huge(1._dp))),'GP kernel theta exceeds exponential range')
    kernel%nodes=self%nodes;pos=0
    do i=1,size(kernel%nodes)
      associate(node=>kernel%nodes(i))
        n=leaf_dimensions(node);values=exp(theta(pos+1:pos+n));j=0
        select case(node%kind)
        case(1,2)
          if(n>0) node%lengths=values
        case(3)
          if(.not.node%fixed(1)) then
            j=j+1;node%first=values(j)
          end if
          if(.not.node%fixed(2)) node%lengths(1)=values(j+1)
        case(4)
          if(.not.node%fixed(1)) then
            j=j+1;node%lengths(1)=values(j)
          end if
          if(.not.node%fixed(2)) node%first=values(j+1)
        case(5:7)
          if(n>0) node%first=values(1)
        end select
        pos=pos+n
      end associate
    end do
  end function
  subroutine leaf_value(node,x,y,auto,i,j,value,gradient)
    type(kernel_node), intent(in) :: node
    real(dp), intent(in) :: x(:),y(:)
    logical, intent(in) :: auto
    integer, intent(in) :: i,j
    real(dp), intent(out) :: value,gradient(:)
    real(dp) :: d(size(x)),lengths(size(x)),r2,r,z,base,tmp,factor,arg,s,c,l,a,nu,logk
    real(dp) :: derivative(size(x))
    integer :: k
    gradient=0;derivative=0
    if(node%kind<=4) then
      call require(size(node%lengths)==1.or.size(node%lengths)==size(x),'GP kernel length count mismatch')
      call require(all(node%lengths>0),'GP kernel lengths must be positive')
      lengths=node%lengths(1)
      if(size(node%lengths)>1) lengths=node%lengths
      d=((x-y)/lengths)**2;r2=sum(d);r=sqrt(r2)
    end if
    select case(node%kind)
    case(1)
      value=exp(-.5_dp*r2);derivative=d*value
    case(2)
      nu=node%nu
      if(.not.ieee_is_finite(nu)) then
        value=exp(-.5_dp*r2);derivative=d*value
      else if(nu==.5_dp) then
        value=exp(-r)
        if(r>0) derivative=d*value/r
      else if(nu==1.5_dp) then
        z=sqrt(3._dp)*r;value=(1+z)*exp(-z);derivative=3*d*exp(-z)
      else if(nu==2.5_dp) then
        z=sqrt(5._dp)*r;value=(1+z+z*z/3)*exp(-z);derivative=(5._dp/3)*d*(1+z)*exp(-z)
      else
        z=sqrt(2*nu)*r
        if(r==0) z=sqrt(2*nu)*epsilon(1._dp)
        logk=log_besselk(nu,z)
        value=exp((1-nu)*log(2._dp)-log_gamma(nu)+nu*log(z)+logk)
        if(r>0.and.size(gradient)>0) then
          factor=z*exp(log_besselk(nu-1,z)-logk);derivative=value*factor*d/r2
        end if
      end if
      if(auto.and.i==j) then
        value=1;derivative=0
      end if
    case(3)
      l=node%lengths(1);a=node%first
      call check_positive(a,'alpha');base=1+r2/(2*a);value=base**(-a);k=0
      if(.not.node%fixed(1).and.size(gradient)>0) then
        k=k+1;gradient(k)=value*(-a*log(base)+r2/(2*base))
      end if
      if(.not.node%fixed(2).and.size(gradient)>0) gradient(k+1)=value*r2/base
    case(4)
      l=node%lengths(1);call check_positive(node%first,'periodicity')
      r=sqrt(sum((x-y)**2));arg=acos(-1._dp)*r/node%first;s=sin(arg);c=cos(arg)
      value=exp(-2*(s/l)**2);k=0
      if(.not.node%fixed(1).and.size(gradient)>0) then
        k=k+1;gradient(k)=4*(s/l)**2*value
      end if
      if(.not.node%fixed(2).and.size(gradient)>0) gradient(k+1)=4*arg*c*s/l**2*value
    case(5)
      value=dot_product(x,y)+node%first**2
      if(size(gradient)>0) gradient=2*node%first**2
    case(6)
      value=node%first
      if(size(gradient)>0) gradient=value
    case(7)
      value=0
      if(auto.and.i==j) value=node%first
      if(size(gradient)>0) gradient=value
    end select
    if(node%kind==1.or.node%kind==2) then
      if(size(gradient)>0) then
        if(size(node%lengths)==1) then
          gradient(1)=sum(derivative)
        else
          gradient=derivative
        end if
      end if
    end if
  end subroutine
  recursive function evaluate(self,node,x,y,auto,want_gradient) result(out)
    class(gp_kernel), intent(in) :: self
    integer, intent(in) :: node
    real(dp), intent(in) :: x(:,:),y(:,:)
    logical, intent(in) :: auto,want_gradient
    type(evaluation) :: out,left,right
    integer :: i,j,n,m,a,b,dims
    real(dp) :: exponent
    n=size(x,1);m=size(y,1)
    associate(current=>self%nodes(node))
      if(current%kind<=7) then
        dims=0;if(want_gradient) dims=leaf_dimensions(current)
        allocate(out%value(n,m),out%gradient(n,m,dims))
        do j=1,m
          do i=1,n
            call leaf_value(current,x(i,:),y(j,:),auto,i,j,out%value(i,j),out%gradient(i,j,:))
          end do
        end do
        return
      end if
      left=evaluate(self,current%left,x,y,auto,want_gradient);a=size(left%gradient,3)
      if(current%kind==10) then
        exponent=current%exponent
        call require(.not.any(left%value<0).or.exponent==anint(exponent),'GP kernel fractional power of negative value')
        call require(.not.any(left%value==0).or.exponent>=0,'GP kernel negative power of zero value')
        out%value=left%value**exponent;allocate(out%gradient(n,m,a));out%gradient=0
        if(exponent/=0) then
          do j=1,m
            do i=1,n
              if(left%value(i,j)/=0) out%gradient(i,j,:)= &
                exponent*left%value(i,j)**(exponent-1)*left%gradient(i,j,:)
            end do
          end do
        end if
        return
      end if
      right=evaluate(self,current%right,x,y,auto,want_gradient);b=size(right%gradient,3)
      allocate(out%gradient(n,m,a+b))
      if(current%kind==8) then
        out%value=left%value+right%value;out%gradient(:,:,:a)=left%gradient;out%gradient(:,:,a+1:)=right%gradient
      else
        out%value=left%value*right%value
        do i=1,a
          out%gradient(:,:,i)=left%gradient(:,:,i)*right%value
        end do
        do i=1,b
          out%gradient(:,:,a+i)=right%gradient(:,:,i)*left%value
        end do
      end if
    end associate
  end function
  function kernel_matrix(self,x,y) result(matrix)
    class(gp_kernel), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:)
    real(dp), allocatable :: matrix(:,:)
    type(evaluation) :: out
    call require(self%is_set(),'GP kernel is empty');call check_x(x)
    if(present(y)) then
      call check_x(y);call require(size(x,2)==size(y,2),'GP kernel feature count mismatch')
      out=evaluate(self,size(self%nodes),x,y,.false.,.false.)
    else
      out=evaluate(self,size(self%nodes),x,x,.true.,.false.)
    end if
    matrix=out%value
  end function
  function kernel_gradient(self,x) result(gradient)
    class(gp_kernel), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: gradient(:,:,:)
    type(evaluation) :: out
    call require(self%is_set(),'GP kernel is empty');call check_x(x)
    out=evaluate(self,size(self%nodes),x,x,.true.,.true.);gradient=out%gradient
  end function
  function kernel_diag(self,x) result(diagonal)
    class(gp_kernel), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: diagonal(:),values(:,:)
    integer :: k,i,n
    call require(self%is_set(),'GP kernel is empty');call check_x(x)
    n=size(x,1);allocate(values(n,size(self%nodes)))
    do k=1,size(self%nodes)
      associate(node=>self%nodes(k))
        select case(node%kind)
        case(1:4);values(:,k)=1
        case(5);values(:,k)=sum(x*x,dim=2)+node%first**2
        case(6,7);values(:,k)=node%first
        case(8);values(:,k)=values(:,node%left)+values(:,node%right)
        case(9);values(:,k)=values(:,node%left)*values(:,node%right)
        case(10)
          call require(.not.any(values(:,node%left)<0).or.node%exponent==anint(node%exponent), &
            'GP kernel fractional power of negative value')
          call require(.not.any(values(:,node%left)==0).or.node%exponent>=0,'GP kernel negative power of zero value')
          values(:,k)=values(:,node%left)**node%exponent
        end select
        if(node%kind<=4) then
          call require(size(node%lengths)==1.or.size(node%lengths)==size(x,2),'GP kernel length count mismatch')
          call require(all(node%lengths>0),'GP kernel lengths must be positive')
        end if
        if(node%kind==3) call check_positive(node%first,'alpha')
        if(node%kind==4) call check_positive(node%first,'periodicity')
      end associate
    end do
    diagonal=values(:,size(self%nodes))
  end function
end module
