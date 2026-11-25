program hamming_loss_invalid_driver
 use sklearn_subset_metrics,only:hamming_loss
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
 case('vector_size');value=hamming_loss([0,1],[0])
 case('weight_length');value=hamming_loss([0,1],[0,1],sample_weight=[1._dp])
 case('zero_weight_normalized');value=hamming_loss([0,1],[0,1],sample_weight=[0._dp,0._dp])
 case('empty_vector_weight');value=hamming_loss([integer::],[integer::],sample_weight=[real(dp)::])
 case('empty_unweighted_multi')
  a%n_rows=0;a%indptr=[1];a%indices=[integer::];a%data=[real(dp)::];value=hamming_loss(a,a)
 case('int64_range');value=hamming_loss([2._dp**63,0._dp],[0,1])
 case('matrix_weight_length');value=hamming_loss(reshape([0,1,0,1],[2,2]), &
  reshape([0,1,0,1],[2,2]),sample_weight=[1._dp])
 case('continuous');value=hamming_loss([0._dp,.5_dp],[0,1])
 case('target_nan');value=hamming_loss([0._dp,ieee_value(0._dp,ieee_quiet_nan)],[0,1])
 case('target_inf');value=hamming_loss([0._dp,ieee_value(0._dp,ieee_positive_inf)],[0,1])
 case('dense_multioutput');value=hamming_loss(reshape([0,1,2,0],[2,2]),reshape([0,1,2,0],[2,2]))
 case('dense_zero_width');value=hamming_loss(empty_width,empty_width)
 case('empty_column');value=hamming_loss(empty_col,empty_col)
 case('mixed_format');value=hamming_loss([0,1],reshape([0,1,0,1],[2,2]))
 case('matrix_shape');value=hamming_loss(reshape([0,1,0,1],[2,2]),reshape([0,1,0,1,0,1],[2,3]))
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
  value=hamming_loss(a,a)
 end select
 print *, value
 error stop 'invalid hamming loss case accepted'
end program
