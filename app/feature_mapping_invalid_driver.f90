program feature_mapping_invalid_driver
  use sklearn_feature_extraction
  use sklearn_murmurhash, only: murmurhash3_32
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(dict_vectorizer) :: vectorizer
  type(feature_hasher) :: hasher
  type(feature_row) :: rows(1),empty(0)
  type(feature_row), allocatable :: inverse(:)
  type(feature_string), allocatable :: names(:)
  type(csr_matrix) :: matrix
  real(dp) :: x(1,1),nan,inf
  integer :: index
  integer(int64) :: hash
  character(40) :: arg
  call get_command_argument(1,arg)
  rows(1)%items=[feature_item(key='a',value=1._dp)]
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf);x=1
  select case(trim(arg))
  case('unfitted_transform');matrix=vectorizer%transform(rows);stop 0
  case('unfitted_names');names=vectorizer%get_feature_names_out();stop 0
  case('unfitted_index');index=vectorizer%vocabulary_index('a');stop 0
  case('unfitted_restrict');call vectorizer%restrict([1]);stop 0
  case('unfitted_inverse');inverse=vectorizer%inverse_transform(x);stop 0
  case('dict_empty_transform');call vectorizer%fit(rows);matrix=vectorizer%transform(empty);stop 0
  case('dict_empty_fit_transform');matrix=vectorizer%fit_transform(empty);stop 0
  case('unallocated_key');deallocate(rows(1)%items(1)%key)
  case('two_categories')
    rows(1)%items(1)%category='x';rows(1)%items(1)%categories=[feature_string('y')]
  case('none_category');rows(1)%items(1)%category='x';rows(1)%items(1)%is_none=.true.
  case('none_list');rows(1)%items(1)%categories=[feature_string('y')];rows(1)%items(1)%is_none=.true.
  case('unallocated_category');allocate(rows(1)%items(1)%categories(1))
  case('dict_tokens');allocate(rows(1)%tokens(0))
  case('hasher_zero_features');hasher%n_features=0;call hasher%fit();stop 0
  case('hasher_negative_features');hasher%n_features=-1;call hasher%fit();stop 0
  case('hasher_mode');hasher%input_type='bad';call hasher%fit();stop 0
  case('hasher_empty');matrix=hasher%transform(empty);stop 0
  case('hasher_none');rows(1)%items(1)%is_none=.true.;matrix=hasher%transform(rows);stop 0
  case('hasher_list');rows(1)%items(1)%categories=[feature_string('x')];matrix=hasher%transform(rows);stop 0
  case('hasher_key');deallocate(rows(1)%items(1)%key);matrix=hasher%transform(rows);stop 0
  case('hasher_tokens');allocate(rows(1)%tokens(0));matrix=hasher%transform(rows);stop 0
  case('string_items');hasher%input_type='string';matrix=hasher%transform(rows);stop 0
  case('unallocated_token')
    hasher%input_type='string';deallocate(rows(1)%items);allocate(rows(1)%tokens(1))
    matrix=hasher%transform(rows);stop 0
  case('negative_seed');hash=murmurhash3_32('a',-1_int64);stop 0
  case('large_seed');hash=murmurhash3_32('a',4294967296_int64);stop 0
  case default
    call vectorizer%fit(rows)
    select case(trim(arg))
    case('mask_width');call vectorizer%restrict([.true.,.false.]);stop 0
    case('index_zero');call vectorizer%restrict([0]);stop 0
    case('index_large');call vectorizer%restrict([2]);stop 0
    case('index_negative');call vectorizer%restrict([-1]);stop 0
    case('index_duplicate');call vectorizer%restrict([1,1]);stop 0
    case('inverse_empty_rows');inverse=vectorizer%inverse_transform(x(:0,:));stop 0
    case('inverse_empty_columns');inverse=vectorizer%inverse_transform(x(:,:0));stop 0
    case('inverse_width');inverse=vectorizer%inverse_transform(reshape([1._dp,2._dp],[1,2]));stop 0
    case('inverse_nan');x=nan;inverse=vectorizer%inverse_transform(x);stop 0
    case('inverse_inf');x=inf;inverse=vectorizer%inverse_transform(x);stop 0
    case('csr_inverse_nan');matrix=vectorizer%transform(rows);matrix%data=nan
    case('csr_inverse_inf');matrix=vectorizer%transform(rows);matrix%data=inf
    case('csr_inverse_width');matrix=vectorizer%transform(rows);matrix%n_cols=2
    case('csr_inverse_empty');matrix=vectorizer%transform(rows);matrix%n_rows=0;matrix%indptr=[1]
      deallocate(matrix%data,matrix%indices);allocate(matrix%data(0),matrix%indices(0))
    case('csr_inverse_invalid');matrix=vectorizer%transform(rows);matrix%indptr=[0,1]
    case default;stop 2
    end select
    inverse=vectorizer%inverse_transform(matrix);stop 0
  end select
  call vectorizer%fit(rows)
end program
