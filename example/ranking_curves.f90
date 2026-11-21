program ranking_curves
  use sklearn, only: dp,roc_curve,precision_recall_curve,det_curve,roc_auc_score,average_precision_score, &
    brier_score_loss,calibration_curve
  implicit none
  integer :: y(4)=[0,0,1,1]
  real(dp) :: scores(4)=[.1_dp,.4_dp,.35_dp,.8_dp]
  real(dp), allocatable :: x(:),a(:),thresholds(:)
  call roc_curve(y,scores,x,a,thresholds)
  print *, 'ROC points: ',size(x),' AUC: ',roc_auc_score(y,scores)
  print *, 'Standardized partial AUC at FPR .5: ',roc_auc_score(y,scores,max_fpr=.5_dp)
  call precision_recall_curve(y,scores,a,x,thresholds)
  print *, 'PR points: ',size(a),' average precision: ',average_precision_score(y,scores)
  call det_curve(y,scores,x,a,thresholds)
  print *, 'DET thresholds: ',thresholds
  print *, 'Brier loss: ',brier_score_loss(y,scores)
  call calibration_curve(y,scores,a,x,n_bins=3)
  print *, 'Calibration observed positive fraction: ',a
  print *, 'Calibration mean probability: ',x
end program
