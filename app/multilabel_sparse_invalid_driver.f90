program multilabel_sparse_invalid_driver
 use sklearn_label_binarization,only:multi_label_binarizer,label_set
 use sklearn_sparse,only:csr_matrix
 use sklearn_common,only:dp
 use,intrinsic::ieee_arithmetic
 implicit none
 type(multi_label_binarizer)::model
 type(label_set)::train(1)
 type(label_set),allocatable::decoded(:)
 type(csr_matrix)::z,a
 character(48)::name
 call get_command_argument(1,name);train(1)%values=[0,1]
 z%n_rows=1;z%n_cols=2;z%indptr=[1,2];z%indices=[1];z%data=[1._dp]
 select case(trim(name))
 case('transform_unfitted');a=model%transform_csr(train)
 case('inverse_unfitted');decoded=model%inverse_transform(z)
 case('duplicate_classes');model%classes=[1,1];a=model%fit_transform_csr(train)
 case default
  call model%fit(train)
  select case(trim(name))
  case('inverse_width');z%n_cols=3
  case('inverse_nonbinary');z%data=2
  case('inverse_nan');z%data=ieee_value(0._dp,ieee_quiet_nan)
  case('inverse_inf');z%data=ieee_value(0._dp,ieee_positive_inf)
  case('csr_unallocated');deallocate(z%indptr,z%indices,z%data)
  case('csr_bad_index');z%indices=3
  case('csr_bad_pointer');z%indptr(2)=4
  case('csr_negative_shape');z%n_rows=-1
  end select
  decoded=model%inverse_transform(z)
 end select
 error stop 'invalid multilabel sparse case accepted'
end program
