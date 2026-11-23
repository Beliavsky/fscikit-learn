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
    procedure :: can_transform=>protocol_true
    procedure :: is_fitted=>protocol_false
    procedure :: bridge_kind=>protocol_zero
    procedure :: bridge_features=>protocol_zero
    procedure :: bridge_supports=>protocol_supports
    procedure :: bridge_classes=>protocol_classes
    procedure :: bridge_predict=>protocol_predict
    procedure :: bridge_labels=>protocol_labels
    procedure :: bridge_matrix=>protocol_matrix
    procedure :: bridge_proba=>protocol_proba
    procedure :: bridge_log_proba=>protocol_proba
    procedure :: bridge_score_samples=>protocol_predict
    procedure :: bridge_score_unsupervised=>protocol_score_unsupervised
    procedure :: bridge_decision=>protocol_decision
    procedure :: bridge_score_real=>protocol_score_real
    procedure :: bridge_score_integer=>protocol_score_integer
    procedure :: bridge_score_matrix=>protocol_score_matrix
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
  logical function protocol_true(self) result(ok)
    class(dense_transformer), intent(in) :: self
    ok=.true.
  end function
  logical function protocol_false(self) result(ok)
    class(dense_transformer), intent(in) :: self
    ok=.false.
  end function
  integer function protocol_zero(self) result(value)
    class(dense_transformer), intent(in) :: self
    value=0
  end function
  recursive logical function protocol_supports(self,name) result(ok)
    class(dense_transformer), intent(in) :: self
    character(*), intent(in) :: name
    ok=.false.
    select case(name)
    case('fit');ok=.true.
    case('transform','fit_transform','get_feature_names_out');ok=self%can_transform()
    case('inverse_transform');ok=self%can_inverse()
    end select
  end function
  function protocol_classes(self) result(value)
    class(dense_transformer), intent(in) :: self
    integer, allocatable :: value(:)
    allocate(value(0));call require(.false.,'native transformer has no classes delegation')
  end function
  function protocol_predict(self,x) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:)
    allocate(value(0));call require(.false.,'native transformer has no predict delegation')
  end function
  function protocol_labels(self,x) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: value(:)
    allocate(value(0));call require(.false.,'native transformer has no labels delegation')
  end function
  function protocol_matrix(self,x) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:)
    allocate(value(0,0));call require(.false.,'native transformer has no matrix delegation')
  end function
  function protocol_proba(self,x) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:)
    allocate(value(0,0));call require(.false.,'native transformer has no proba delegation')
  end function
  function protocol_decision(self,x) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:)
    allocate(value(0,0));call require(.false.,'native transformer has no decision delegation')
  end function
  real(dp) function protocol_score_real(self,x,y,sample_weight) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=0;call require(.false.,'native transformer has no score delegation')
  end function
  real(dp) function protocol_score_integer(self,x,y,sample_weight) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=0;call require(.false.,'native transformer has no score delegation')
  end function
  real(dp) function protocol_score_matrix(self,x,y,sample_weight) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=0;call require(.false.,'native transformer has no score delegation')
  end function
  real(dp) function protocol_score_unsupervised(self,x) result(value)
    class(dense_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    value=0;call require(.false.,'native transformer has no unsupervised score delegation')
  end function
end module
