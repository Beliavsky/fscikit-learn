program categorical_scores_invalid_driver
 use sklearn_categorical_scores,only:balanced_accuracy_score,matthews_corrcoef,cohen_kappa_score
 use sklearn_common,only:dp
 use,intrinsic::ieee_arithmetic
 implicit none
 character(48)::mode
 real(dp)::value,bad
 integer::i
 call get_command_argument(1,mode)
 select case(trim(mode))
 case('balanced_length');value=balanced_accuracy_score([0,1],[0])
 case('mcc_length');value=matthews_corrcoef([0,1],[0])
 case('kappa_length');value=cohen_kappa_score([0,1],[0])
 case('balanced_weights');value=balanced_accuracy_score([0,1],[0,1],sample_weight=[1._dp])
 case('mcc_weights');value=matthews_corrcoef([0,1],[0,1],sample_weight=[1._dp])
 case('kappa_weights');value=cohen_kappa_score([0,1],[0,1],sample_weight=[1._dp])
 case('continuous');value=balanced_accuracy_score([0._dp,.5_dp],[0._dp,1._dp])
 case('target_nan')
  bad=ieee_value(0._dp,ieee_quiet_nan);value=matthews_corrcoef([bad,1._dp],[0._dp,1._dp])
 case('target_inf')
  bad=ieee_value(0._dp,ieee_positive_inf);value=cohen_kappa_score([bad,1._dp],[0._dp,1._dp])
 case('int64_range');value=balanced_accuracy_score([2._dp**63,0._dp],[0._dp,1._dp])
 case('matrix_width');value=matthews_corrcoef(reshape([0,1,0,1],[2,2]),[0,1])
 case('empty_column');value=cohen_kappa_score(reshape([integer::],[0,1]),[integer::])
 case('labels_empty');value=cohen_kappa_score([0,1],[0,1],labels=[integer::])
 case('labels_missing');value=cohen_kappa_score([0,1],[0,1],labels=[2,3])
 case('label_kind');value=cohen_kappa_score([0,1],[0,1],labels=[0.,1.])
 case('kappa_weight_mode');value=cohen_kappa_score([0,1],[0,1],weights='cubic')
 case('adjusted_no_recalls');value=balanced_accuracy_score([0,1],[0,1],sample_weight=[0._dp,0._dp],adjusted=.true.)
 case('adjusted_empty');value=balanced_accuracy_score([integer::],[integer::],adjusted=.true.)
 case('kappa_empty_weights');value=cohen_kappa_score([integer::],[integer::],sample_weight=[1._dp])
 case('mcc_empty_weights');value=matthews_corrcoef([integer::],[integer::],sample_weight=[1._dp])
 case('balanced_empty_weights');value=balanced_accuracy_score([integer::],[integer::],sample_weight=[1._dp])
 case('matrix_capacity');value=cohen_kappa_score([0],[0],labels=[(i,i=0,49999)])
 case default
  error stop 'unknown categorical score invalid case'
 end select
 print *,value
 error stop 'invalid categorical score case accepted'
end program
