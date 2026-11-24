program subset_metrics_invalid_driver
 use sklearn_subset_metrics,only:accuracy_score,zero_one_loss
 use sklearn_sparse,only:csr_matrix
 use sklearn_common,only:dp
 use,intrinsic::ieee_arithmetic
 implicit none
 character(48)::name
 type(csr_matrix)::a
 real(dp)::value,empty_col(0,1),empty_width(2,0)
 call get_command_argument(1,name)
 a%n_rows=2;a%n_cols=2;a%indptr=[1,2,3];a%indices=[1,2];a%data=[1._dp,1._dp]
 select case(trim(name))
 case('vector_size');value=accuracy_score([0,1],[0])
 case('weight_length');value=accuracy_score([0,1],[0,1],sample_weight=[1._dp])
 case('zero_weight_normalized');value=accuracy_score([0,1],[0,1],sample_weight=[0._dp,0._dp])
 case('zero_loss_zero_weight');value=zero_one_loss([0,1],[0,1],sample_weight=[0._dp,0._dp])
 case('continuous');value=accuracy_score([0._dp,.5_dp],[0,1])
 case('target_nan');value=accuracy_score([0._dp,ieee_value(0._dp,ieee_quiet_nan)],[0,1])
 case('target_inf');value=accuracy_score([0._dp,ieee_value(0._dp,ieee_positive_inf)],[0,1])
 case('dense_multioutput');value=accuracy_score(reshape([0,1,2,0],[2,2]),reshape([0,1,2,0],[2,2]))
 case('dense_zero_width');value=accuracy_score(empty_width,empty_width)
 case('empty_column');value=accuracy_score(empty_col,empty_col)
 case('mixed_format');value=accuracy_score([0,1],reshape([0,1,0,1],[2,2]))
 case('matrix_shape');value=accuracy_score(reshape([0,1,0,1],[2,2]),reshape([0,1,0,1,0,1],[2,3]))
 case default
  select case(trim(name))
  case('csr_column');a%n_cols=1;a%indices=1
  case('csr_nonindicator');a%data=[2._dp,3._dp]
  case('csr_fractional');a%data=.5_dp
  case('csr_nan');a%data=ieee_value(0._dp,ieee_quiet_nan)
  case('csr_inf');a%data=ieee_value(0._dp,ieee_positive_inf)
  case('csr_unallocated');deallocate(a%indptr,a%indices,a%data)
  case('csr_bad_index');a%indices=3
  case('csr_bad_pointer');a%indptr(3)=5
  case('csr_negative_shape');a%n_rows=-1
  end select
  value=accuracy_score(a,a)
 end select
 print *, value
 error stop 'invalid subset metric case accepted'
end program
