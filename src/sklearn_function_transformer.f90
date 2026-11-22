! FunctionTransformer contracts: sklearn 1.6.1 preprocessing/_function_transformer.py.
! BSD-3-Clause attribution is retained in THIRD_PARTY_NOTICES.md.
module sklearn_function_transformer
  use sklearn_common, only: dp,require,check_x
  use sklearn_feature_extraction, only: feature_string
  use sklearn_ieee_math, only: ieee_difference,ieee_product
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  private
  public :: function_transformer,matrix_function,matrices_close
  abstract interface
    function matrix_function(x) result(z)
      import dp
      real(dp), intent(in) :: x(:,:)
      real(dp), allocatable :: z(:,:)
    end function
    function name_function(transformer,input_features) result(names)
      import feature_string
      class(*), intent(in) :: transformer
      type(feature_string), intent(in) :: input_features(:)
      type(feature_string), allocatable :: names(:)
    end function
  end interface
  type :: function_transformer
    procedure(matrix_function), pointer, nopass :: func=>null(),inverse_func=>null()
    procedure(name_function), pointer, nopass :: names_func=>null()
    logical :: validate=.false.,check_inverse=.true.,inverse_checked_=.false.,inverse_consistent_=.true.
    integer :: n_features_in_=0
    character(16) :: feature_names_out='none'
    type(feature_string), allocatable :: feature_names_in_(:)
  contains
    procedure :: fit=>function_fit
    procedure :: transform=>function_forward
    procedure :: inverse_transform=>function_inverse
    procedure :: fit_transform=>function_fit_transform
    procedure :: get_feature_names_out=>function_names
  end type
contains
  logical function matrices_close(a,b,rtol,atol) result(close)
    real(dp), intent(in) :: a(:,:),b(:,:),rtol,atol
    integer :: dims(2),i,j,ia,ja,ib,jb,k
    real(dp) :: av,bv,difference,tolerance
    do k=1,2
      if(size(a,k)==size(b,k)) then
        dims(k)=size(a,k)
      else if(size(a,k)==1) then
        dims(k)=size(b,k)
      else if(size(b,k)==1) then
        dims(k)=size(a,k)
      else
        call require(.false.,'inverse check arrays cannot be broadcast')
      end if
    end do
    close=.false.
    do j=1,dims(2)
      ja=min(j,size(a,2));jb=min(j,size(b,2))
      do i=1,dims(1)
        ia=min(i,size(a,1));ib=min(i,size(b,1));av=a(ia,ja);bv=b(ib,jb)
        if(ieee_is_nan(av).or.ieee_is_nan(bv)) return
        if(av==bv) cycle
        if(.not.ieee_is_finite(av).or..not.ieee_is_finite(bv)) return
        difference=abs(ieee_difference(av,bv))
        tolerance=ieee_difference(atol,-ieee_product(rtol,abs(bv)))
        if(difference>tolerance) return
      end do
    end do
    close=.true.
  end function
  subroutine controls(self)
    class(function_transformer), intent(in) :: self
    call require(any(self%feature_names_out==['none            ','one-to-one      ']), &
      'FunctionTransformer invalid feature_names_out')
    call require(.not.(self%feature_names_out=='one-to-one'.and.associated(self%names_func)), &
      'FunctionTransformer choose one-to-one or a names callback')
  end subroutine
  subroutine function_fit(self,x,feature_names)
    class(function_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:),roundtrip(:,:)
    integer :: i,step
    call controls(self)
    if(self%validate) call check_x(x)
    self%n_features_in_=size(x,2)
    if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
    if(present(feature_names)) then
      call require(size(feature_names)==size(x,2),'FunctionTransformer fit name count mismatch')
      do i=1,size(feature_names)
        call require(allocated(feature_names(i)%text),'FunctionTransformer feature name is unallocated')
      end do
      self%feature_names_in_=feature_names
    end if
    self%inverse_checked_=.false.;self%inverse_consistent_=.true.
    if(self%check_inverse.and.associated(self%func).and.associated(self%inverse_func)) then
      step=max(1,size(x,1)/100)
      z=self%transform(x(1:size(x,1):step,:));roundtrip=self%inverse_transform(z)
      self%inverse_checked_=.true.
      self%inverse_consistent_=matrices_close(x(1:size(x,1):step,:),roundtrip,1.e-7_dp,1.e-9_dp)
    end if
  end subroutine
  function function_forward(self,x) result(z)
    class(function_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call controls(self)
    if(self%validate) then
      call check_x(x)
      if(self%n_features_in_>0) call require(size(x,2)==self%n_features_in_, &
        'FunctionTransformer transform feature width mismatch')
    end if
    if(associated(self%func)) then
      z=self%func(x)
    else
      z=x
    end if
  end function
  function function_inverse(self,x) result(z)
    class(function_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call controls(self)
    ! Source inverse validates input but does not enforce fitted forward width.
    if(self%validate) call check_x(x)
    if(associated(self%inverse_func)) then
      z=self%inverse_func(x)
    else
      z=x
    end if
  end function
  function function_fit_transform(self,x,feature_names) result(z)
    class(function_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  function function_names(self,input_features) result(names)
    class(function_transformer), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:),input(:)
    integer :: i
    character(32) :: buffer
    call controls(self)
    call require(self%feature_names_out=='one-to-one'.or.associated(self%names_func), &
      'FunctionTransformer output feature names are not configured')
    if(present(input_features)) then
      if(self%n_features_in_>0) call require(size(input_features)==self%n_features_in_, &
        'FunctionTransformer input name count mismatch')
      do i=1,size(input_features)
        call require(allocated(input_features(i)%text),'FunctionTransformer feature name is unallocated')
      end do
      input=input_features
      if(allocated(self%feature_names_in_)) then
        call require(size(input)==size(self%feature_names_in_),'FunctionTransformer stored name count mismatch')
        do i=1,size(input)
          call require(len(input(i)%text)==len(self%feature_names_in_(i)%text), &
            'FunctionTransformer input feature names differ')
          call require(input(i)%text==self%feature_names_in_(i)%text,'FunctionTransformer input feature names differ')
        end do
      end if
    else if(allocated(self%feature_names_in_)) then
      input=self%feature_names_in_
    else if(self%n_features_in_>0) then
      allocate(input(self%n_features_in_))
      do i=1,size(input)
        write(buffer,'(a,i0)') 'x',i-1;input(i)%text=trim(buffer)
      end do
    else
      call require(associated(self%names_func),'FunctionTransformer names require fit validation or input names')
      allocate(input(0))
    end if
    if(associated(self%names_func)) then
      names=self%names_func(self,input)
    else
      names=input
    end if
  end function
end module
