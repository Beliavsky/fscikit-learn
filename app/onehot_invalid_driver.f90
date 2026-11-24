program onehot_invalid_driver
 use sklearn_common,only:dp
 use sklearn_one_hot_encoding,only:one_hot_encoder,onehot_inverse_result,onehot_name_value
 use sklearn_feature_extraction,only:feature_string
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::ieee_arithmetic
 implicit none
 type(one_hot_encoder)::model
 type(onehot_inverse_result)::decoded
 type(feature_string),allocatable::names(:)
 type(csr_matrix)::s
 real(dp)::x(4,2),qnan
 real(dp),allocatable::z(:,:),values(:,:)
 character(64)::case
 x(:,1)=[0._dp,1._dp,1._dp,2._dp];x(:,2)=[0._dp,0._dp,1._dp,1._dp]
 qnan=ieee_value(0._dp,ieee_quiet_nan);call get_command_argument(1,case)
 select case(trim(case))
 case('empty_rows');call model%fit(x(:0,:))
 case('empty_features');call model%fit(x(:,:0))
 case('fit_inf');x(1,1)=ieee_value(0._dp,ieee_positive_inf);call model%fit(x)
 case('unknown_mode');model%handle_unknown='bad';call model%fit(x)
 case('drop_mode');model%drop='bad';call model%fit(x)
 case('count_negative');model%min_frequency_count=-1;call model%fit(x)
 case('fraction_nan');model%min_frequency_fraction=qnan;call model%fit(x)
 case('fraction_negative');model%min_frequency_fraction=-.1_dp;call model%fit(x)
 case('fraction_one');model%min_frequency_fraction=1;call model%fit(x)
 case('both_frequencies');model%min_frequency_count=2;model%min_frequency_fraction=.2_dp;call model%fit(x)
 case('maximum_negative');model%max_categories=-1;call model%fit(x)
 case('category_width');allocate(model%categories(1));call model%fit(x)
 case('category_unallocated');allocate(model%categories(2));call model%fit(x)
 case('category_empty')
  allocate(model%categories(2));model%categories(1)%values=[real(dp)::];call model%fit(x)
 case('category_duplicate')
  allocate(model%categories(2));model%categories(1)%values=[0._dp,0._dp];call model%fit(x)
 case('category_unsorted')
  allocate(model%categories(2));model%categories(1)%values=[1._dp,0._dp];call model%fit(x)
 case('category_nan_order')
  allocate(model%categories(2));model%categories(1)%values=[qnan,0._dp];call model%fit(x)
 case('unknown_fit')
  allocate(model%categories(2));model%categories(1)%values=[0._dp,1._dp];call model%fit(x)
 case('drop_width');model%drop_values=[0._dp];call model%fit(x)
 case('drop_unknown');model%drop_values=[9._dp,0._dp];call model%fit(x)
 case('drop_infrequent');model%drop_values=[0._dp,0._dp];model%min_frequency_count=2;call model%fit(x)
 case('fit_name_count');allocate(names(1));names(1)%text='one';call model%fit(x,names)
 case('fit_name_unallocated');allocate(names(2));call model%fit(x,names)
 case('transform_unfitted');z=model%transform(x)
 case('inverse_unfitted');decoded=model%inverse_transform(x)
 case('names_unfitted');names=model%get_feature_names_out()
 case('result_unallocated');values=decoded%as_numeric()
 case('query_width');call model%fit(x);z=model%transform(x(:,:1))
 case('query_empty');call model%fit(x);z=model%transform(x(:0,:))
 case('query_inf');call model%fit(x);x(1,1)=ieee_value(0._dp,ieee_positive_inf);z=model%transform(x)
 case('query_unknown');call model%fit(x);z=model%transform(x+10)
 case('query_name_count')
  call model%fit(x);allocate(names(1));names(1)%text='one';z=model%transform(x,names)
 case('query_name_unallocated');call model%fit(x);allocate(names(2));names=model%get_feature_names_out(names)
 case('query_name_difference')
  allocate(names(2));names(1)%text='one';names(2)%text='two';call model%fit(x,names)
  names(2)%text='bad';z=model%transform(x,names)
 case('inverse_width');call model%fit(x);decoded=model%inverse_transform(x)
 case('inverse_empty');call model%fit(x);decoded=model%inverse_transform(x(:0,:))
 case('inverse_nan');call model%fit(x);z=model%transform(x);z(1,1)=qnan;decoded=model%inverse_transform(z)
 case('inverse_inf')
  call model%fit(x);z=model%transform(x);z(1,1)=ieee_value(0._dp,ieee_positive_inf);decoded=model%inverse_transform(z)
 case('inverse_zero_sum');call model%fit(x);z=model%transform(x)*0;decoded=model%inverse_transform(z)
 case('inverse_broadcast')
  model%drop='if_binary';call model%fit(x);z=model%transform(x(:1,:))*0;decoded=model%inverse_transform(z)
 case('inverse_zero_width')
  model%drop='first';x=1;call model%fit(x);z=model%transform(x);decoded=model%inverse_transform(z)
 case('result_none')
  model%handle_unknown='ignore';call model%fit(x);z=model%transform(x)*0
  decoded=model%inverse_transform(z);values=decoded%as_numeric()
 case('result_infrequent')
  model%min_frequency_count=2;call model%fit(x);z=model%transform(x)
  decoded=model%inverse_transform(z);values=decoded%as_numeric()
 case('bad_callback')
  model%feature_name_combiner=>bad_callback;call model%fit(x);names=model%get_feature_names_out()
 case('csr_bad_index')
  call model%fit(x);s=model%transform_csr(x);s%indices(1)=0;decoded=model%inverse_transform(s)
 case default
  error stop 'unknown onehot invalid case'
 end select
 error stop 'onehot invalid case was accepted'
contains
 function bad_callback(feature,category) result(text)
  character(*),intent(in)::feature
  type(onehot_name_value),intent(in)::category
  type(feature_string)::text
 end function
end program
