program likelihood_ratios_invalid_driver
 use sklearn_likelihood_ratios,only:class_likelihood_ratios
 use sklearn_common,only:dp
 use,intrinsic::ieee_arithmetic
 implicit none
 character(40)::mode
 real(dp)::ratios(2),bad
 call get_command_argument(1,mode)
 select case(trim(mode))
 case('target_length');ratios=class_likelihood_ratios([0,1],[0])
 case('continuous');ratios=class_likelihood_ratios([0._dp,.5_dp],[0._dp,1._dp])
 case('target_nan')
  bad=ieee_value(0._dp,ieee_quiet_nan);ratios=class_likelihood_ratios([bad,1._dp],[0._dp,1._dp])
 case('target_inf')
  bad=ieee_value(0._dp,ieee_positive_inf);ratios=class_likelihood_ratios([bad,1._dp],[0._dp,1._dp])
 case('int64_range');ratios=class_likelihood_ratios([2._dp**63,0._dp],[0._dp,1._dp])
 case('multiclass');ratios=class_likelihood_ratios([0,1,2],[0,1,2])
 case('matrix_width');ratios=class_likelihood_ratios(reshape([0,1,0,1],[2,2]),[0,1])
 case('empty_column');ratios=class_likelihood_ratios(reshape([integer::],[0,1]),[integer::])
 case('empty_target');ratios=class_likelihood_ratios([integer::],[integer::])
 case('labels_empty');ratios=class_likelihood_ratios([0,1],[0,1],labels=[integer::])
 case('labels_missing');ratios=class_likelihood_ratios([0,1],[0,1],labels=[2,3])
 case('labels_width');ratios=class_likelihood_ratios([0,1],[0,1],labels=[0,1,2])
 case('label_kind');ratios=class_likelihood_ratios([0,1],[0,1],labels=[0.,1.])
 case('weight_length');ratios=class_likelihood_ratios([0,1],[0,1],sample_weight=[1._dp])
 case default
  error stop 'unknown likelihood ratio invalid case'
 end select
 print *,ratios
 error stop 'invalid likelihood ratio case accepted'
end program
