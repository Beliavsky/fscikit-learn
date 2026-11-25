program confusion_matrix_invalid_driver
 use sklearn_confusion_matrix,only:confusion_matrix
 use sklearn_common,only:dp
 use,intrinsic::ieee_arithmetic
 implicit none
 character(48)::mode
 integer,allocatable::counts(:,:)
 real(dp),allocatable::values(:,:)
 real(dp)::bad
 integer::i
 call get_command_argument(1,mode)
 select case(trim(mode))
 case('counts_length');counts=confusion_matrix([0,1],[0])
 case('weighted_length');values=confusion_matrix([0,1],[0],sample_weight=[1._dp,1._dp])
 case('normalized_length');values=confusion_matrix([0,1],[0],normalize='true')
 case('weights_length');values=confusion_matrix([0,1],[0,1],sample_weight=[1._dp])
 case('normalized_weights');values=confusion_matrix([0,1],[0,1],sample_weight=[1._dp],normalize='all')
 case('empty_auto_weights');values=confusion_matrix([integer::],[integer::],sample_weight=[1._dp])
 case('empty_auto_normalized_weights')
  values=confusion_matrix([integer::],[integer::],sample_weight=[1._dp],normalize='pred')
 case('labels_empty');counts=confusion_matrix([0,1],[0,1],labels=[integer::])
 case('labels_missing');values=confusion_matrix([0,1],[0,1],labels=[2,3],sample_weight=[1._dp,1._dp])
 case('label_kind');values=confusion_matrix([0,1],[0,1],labels=[0.,1.],normalize='true')
 case('continuous');counts=confusion_matrix([0._dp,.5_dp],[0._dp,1._dp])
 case('target_nan')
  bad=ieee_value(0._dp,ieee_quiet_nan)
  values=confusion_matrix([bad,1._dp],[0._dp,1._dp],sample_weight=[1._dp,1._dp])
 case('target_inf')
  bad=ieee_value(0._dp,ieee_positive_inf);values=confusion_matrix([bad,1._dp],[0._dp,1._dp],normalize='pred')
 case('int64_range');counts=confusion_matrix([2._dp**63,0._dp],[0._dp,1._dp])
 case('matrix_width');counts=confusion_matrix(reshape([0,1,0,1],[2,2]),[0,1])
 case('empty_column');counts=confusion_matrix(reshape([integer::],[0,1]),[integer::])
 case('zero_width');counts=confusion_matrix(reshape([integer::],[2,0]),[0,1])
 case('normalize_mode');values=confusion_matrix([0,1],[0,1],normalize='bad')
 case('weighted_normalize_mode');values=confusion_matrix([0,1],[0,1],sample_weight=[1._dp,1._dp],normalize='bad')
 case('matrix_capacity');counts=confusion_matrix([0],[0],labels=[(i,i=0,49999)])
 case default
  error stop 'unknown confusion matrix invalid case'
 end select
 print *, 'unexpected normal return'
 error stop 'invalid confusion matrix case accepted'
end program
