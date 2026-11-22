! Internal typed bridge for recursive dense compositions; original MIT code.
module sklearn_transformer_protocol
  use sklearn_common, only: dp,require
  use sklearn_feature_extraction, only: feature_string
  implicit none
  private
  public :: dense_transformer
  type, abstract :: dense_transformer
  contains
    procedure(fit_interface), deferred :: fit_data
    procedure(map_interface), deferred :: map_data
    procedure(names_interface), deferred :: feature_names
    procedure :: fit_map_data=>protocol_fit_map
    procedure :: inverse_data=>protocol_inverse
    procedure :: can_inverse=>protocol_no_inverse
  end type
  abstract interface
    subroutine fit_interface(self,x,feature_names)
      import dense_transformer,dp,feature_string
      class(dense_transformer), intent(inout) :: self
      real(dp), intent(in) :: x(:,:)
      type(feature_string), optional, intent(in) :: feature_names(:)
    end subroutine
    function map_interface(self,x) result(z)
      import dense_transformer,dp
      class(dense_transformer), intent(in) :: self
      real(dp), intent(in) :: x(:,:)
      real(dp), allocatable :: z(:,:)
    end function
    function names_interface(self,input_features) result(names)
      import dense_transformer,feature_string
      class(dense_transformer), intent(in) :: self
      type(feature_string), optional, intent(in) :: input_features(:)
      type(feature_string), allocatable :: names(:)
    end function
  end interface
contains
  recursive function protocol_fit_map(self,x,feature_names) result(z)
    class(dense_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit_data(x,feature_names);z=self%map_data(x)
  end function
  function protocol_inverse(self,x) result(z)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    allocate(z(0,0));call require(.false.,'dense transformer has no inverse_transform')
  end function
  logical function protocol_no_inverse(self) result(available)
    class(dense_transformer), intent(in) :: self
    available=.false.
  end function
end module
