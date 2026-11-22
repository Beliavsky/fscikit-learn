program composition_invalid_driver
  use sklearn_common, only: dp,require
  use sklearn_feature_composition
  use sklearn_function_transformer, only: function_transformer
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_feature_extraction, only: feature_string
  implicit none
  type(column_transformer) :: column
  type(feature_union) :: union
  type(function_transformer) :: ft
  type(standard_scaler) :: scaler
  type(feature_string) :: names(3)
  type(feature_string), allocatable :: output(:)
  real(dp) :: x(4,3)
  real(dp), allocatable :: z(:,:)
  integer :: idx
  integer, allocatable :: indices(:)
  character(64) :: argument
  call get_command_argument(1,argument)
  x=reshape([1._dp,2._dp,3._dp,4._dp,2._dp,4._dp,3._dp,5._dp,1._dp,7._dp,4._dp,6._dp],[4,3])
  names(1)%text='a';names(2)%text='b';names(3)%text='c'
  allocate(column%transformers(1));column%transformers(1)%name='main';column%transformers(1)%columns=[1,2]
  allocate(column%transformers(1)%model,source=scaler)
  allocate(union%transformer_list(1));union%transformer_list(1)%name='main'
  allocate(union%transformer_list(1)%model,source=scaler)
  select case(trim(argument))
  case('column_unfitted')
    z=column%transform(x)
  case('column_names_unfitted')
    output=column%get_feature_names_out()
  case('column_index_unfitted')
    idx=column%named_transformer_index('main')
  case('union_unfitted')
    z=union%transform(x)
  case('union_names_unfitted')
    output=union%get_feature_names_out()
  case('name_unallocated')
    deallocate(column%transformers(1)%name);call column%fit(x)
  case('name_dupe')
    call extra_branch();column%transformers(2)%name='main';call column%fit(x)
  case('name_delimiter')
    column%transformers(1)%name='main__bad';call column%fit(x)
  case('name_reserved')
    column%transformers(1)%name='remainder';call column%fit(x)
  case('union_name_reserved')
    union%transformer_list(1)%name='transformer_list';call union%fit(x)
  case('action_invalid')
    column%transformers(1)%action='invalid';call column%fit(x)
  case('model_unallocated')
    deallocate(column%transformers(1)%model);call column%fit(x)
  case('model_unsupported')
    deallocate(column%transformers(1)%model);allocate(column%transformers(1)%model,source=1);call column%fit(x)
  case('empty_unsupported')
    deallocate(column%transformers(1)%model);allocate(column%transformers(1)%model,source=1)
    column%transformers(1)%columns=[integer::];call column%fit(x)
  case('remainder_action')
    column%remainder%action='invalid';call column%fit(x)
  case('remainder_model')
    column%remainder%action='transform';call column%fit(x)
  case('selector_none')
    deallocate(column%transformers(1)%columns);call column%fit(x)
  case('selector_multiple')
    column%transformers(1)%mask=[.true.,.false.,.false.];call column%fit(x)
  case('selector_mask_width')
    deallocate(column%transformers(1)%columns)
    column%transformers(1)%mask=[.true.];call column%fit(x)
  case('index_zero')
    column%transformers(1)%columns=[0];call column%fit(x)
  case('index_negative')
    column%transformers(1)%columns=[-1];call column%fit(x)
  case('index_large')
    column%transformers(1)%columns=[4];call column%fit(x)
  case('named_no_names')
    deallocate(column%transformers(1)%columns)
    column%transformers(1)%column_names=names(:1);call column%fit(x)
  case('named_unallocated')
    deallocate(column%transformers(1)%columns)
    allocate(column%transformers(1)%column_names(1));call column%fit(x,names)
  case('named_missing')
    deallocate(column%transformers(1)%columns)
    column%transformers(1)%column_names=[feature_string("missing")];call column%fit(x,names)
  case('named_dupe')
    deallocate(column%transformers(1)%columns)
    column%transformers(1)%column_names=names(:1);names(2)%text="a";call column%fit(x,names)
  case('callback_unallocated')
    deallocate(column%transformers(1)%columns)
    column%transformers(1)%selector=>empty_selector;call column%fit(x)
  case('callback_bounds')
    deallocate(column%transformers(1)%columns)
    column%transformers(1)%selector=>bad_selector;call column%fit(x)
  case('fit_names_count')
    call column%fit(x,names(:2))
  case('fit_name_unallocated')
    deallocate(names(1)%text);call column%fit(x,names)
  case('query_width')
    call column%fit(x);z=column%transform(x(:,:2))
  case('query_name_count')
    call column%fit(x,names);z=column%transform(x,names(:2))
  case('query_name_missing')
    call column%fit(x,names);names(1)%text="missing";z=column%transform(x,names)
  case('query_name_duplicate')
    call column%fit(x,names);names(3)%text="a";z=column%transform(x,names)
  case('get_names_count')
    call column%fit(x);output=column%get_feature_names_out(names(:2))
  case('get_names_differ')
    call column%fit(x,names);names(1)%text="z";output=column%get_feature_names_out(names)
  case('get_names_length')
    call column%fit(x,names);names(1)%text="a ";output=column%get_feature_names_out(names)
  case('output_unknown')
    call column%fit(x);indices=column%output_indices("missing")
  case('output_unknown_blanks')
    call column%fit(x);indices=column%output_indices("remainder ")
  case('output_duplicates')
    call extra_branch();column%verbose_feature_names_out=.false.
    call column%fit(x);output=column%get_feature_names_out()
  case('output_unconfigured')
    deallocate(column%transformers(1)%model)
    allocate(column%transformers(1)%model,source=ft);call column%fit(x);output=column%get_feature_names_out()
  case('branch_row_mismatch')
    call extra_branch();ft%func=>trim_row
    deallocate(column%transformers(1)%model);allocate(column%transformers(1)%model,source=ft);call column%fit(x)
  case('union_row_mismatch')
    deallocate(union%transformer_list);allocate(union%transformer_list(2))
    union%transformer_list(1)%name="trim";ft%func=>trim_row
    allocate(union%transformer_list(1)%model,source=ft)
    union%transformer_list(2)%name="pass";union%transformer_list(2)%action="passthrough";z=union%fit_transform(x)
  case default
    call require(.false.,'unknown invalid case')
  end select
  error stop 'invalid case unexpectedly accepted'
contains
  subroutine extra_branch()
    type(transformer_branch) :: branch
    branch%name='raw';branch%action='passthrough';branch%columns=[1,2]
    column%transformers=[column%transformers,branch]
  end subroutine
  subroutine empty_selector(a,names,indices)
    real(dp), intent(in) :: a(:,:)
    type(feature_string), intent(in) :: names(:)
    integer, allocatable, intent(out) :: indices(:)
  end subroutine
  subroutine bad_selector(a,names,indices)
    real(dp), intent(in) :: a(:,:)
    type(feature_string), intent(in) :: names(:)
    integer, allocatable, intent(out) :: indices(:)
    indices=[100]
  end subroutine
  function trim_row(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a(:size(a,1)-1,:)
  end function
end program
