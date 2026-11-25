program variance_scores_invalid_driver
 use sklearn,only:dp,r2_score,explained_variance_score
 use,intrinsic::ieee_arithmetic
 implicit none
 character(40)::case
 real(dp)::y(2,2),p(2,2),v,nan,inf
 call get_command_argument(1,case)
 y=reshape([1._dp,2._dp,3._dp,4._dp],[2,2]);p=y
 nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 select case(trim(case))
 case('empty_rows')
  v=r2_score(y(:0,:),p(:0,:))
 case('empty_outputs')
  v=explained_variance_score(y(:,:0),p(:,:0))
 case('row_shape')
  v=r2_score(y,p(:1,:))
 case('column_shape')
  v=explained_variance_score(y,p(:,:1))
 case('mixed_shape')
  v=r2_score(y(:,1),p)
 case('target_nan')
  y(1,1)=nan;v=r2_score(y,p)
 case('prediction_inf')
  p(1,1)=inf;v=explained_variance_score(y,p)
 case('weight_length')
  v=r2_score(y,p,sample_weight=[1._dp])
 case('empty_weights')
  v=explained_variance_score(y,p,sample_weight=[real(dp)::])
 case('zero_weight')
  v=r2_score(y,p,sample_weight=[1._dp,-1._dp])
 case('singleton_ev_zero_weight')
  v=explained_variance_score(y(:1,:),p(:1,:),sample_weight=[0._dp])
 case('output_length')
  v=r2_score(y,p,multioutput=[1._dp])
 case('empty_output_weights')
  v=explained_variance_score(y,p,multioutput=[real(dp)::])
 case('single_output_custom')
  v=r2_score(y(:,1),p(:,1),multioutput=[1._dp])
 case('output_nan')
  v=r2_score(y,p,multioutput=[nan,1._dp])
 case('output_inf')
  v=explained_variance_score(y,p,multioutput=[1._dp,inf])
 case('zero_output_weight')
  v=r2_score(y,p,multioutput=[1._dp,-1._dp])
 case('bad_mode')
  v=explained_variance_score(y,p,multioutput='average')
 case('raw_scalar')
  v=r2_score(y,p,multioutput='raw_values')
 case('singleton_bad_output_length')
  v=r2_score(y(:1,:),p(:1,:),multioutput=[1._dp])
 case default
  error stop 'unknown variance diagnostic'
 end select
 print *,v
 error stop 'invalid variance score case accepted'
end program
