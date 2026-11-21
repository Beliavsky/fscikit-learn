program test_ranking
  use sklearn, only: dp,roc_curve,precision_recall_curve,det_curve,auc,roc_auc_score,average_precision_score, &
    brier_score_loss,calibration_curve
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  integer :: y(4)=[0,0,1,1],i
  real(dp) :: scores(4)=[.1_dp,.4_dp,.35_dp,.8_dp]
  real(dp), allocatable :: a(:),b(:),thresholds(:),large_scores(:)
  integer, allocatable :: large_y(:)
  call roc_curve(y,scores,a,b,thresholds,drop_intermediate=.false.)
  call require(size(a)==5.and..not.ieee_is_finite(thresholds(1)),'ROC origin and infinite threshold')
  call require(all(abs(a-[0._dp,0._dp,.5_dp,.5_dp,1._dp])<1.e-14_dp),'analytic ROC false positives')
  call require(all(abs(b-[0._dp,.5_dp,.5_dp,1._dp,1._dp])<1.e-14_dp),'analytic ROC true positives')
  call require(abs(auc(a,b)-.75_dp)<1.e-14_dp,'trapezoidal ROC area')
  call require(abs(roc_auc_score(y,scores,max_fpr=.5_dp)-2._dp/3)<1.e-14_dp,'standardized partial ROC AUC')
  call require(abs(average_precision_score(y,scores)-5._dp/6)<1.e-14_dp,'step integral average precision')
  call precision_recall_curve(y,scores,a,b,thresholds)
  call require(size(a)==size(thresholds)+1.and.a(size(a))==1.and.b(size(b))==0,'PR terminal point')
  call require(all(thresholds==[.1_dp,.35_dp,.4_dp,.8_dp]),'ascending PR thresholds')
  call det_curve(y,scores,a,b,thresholds)
  call require(all(abs(a-[.5_dp,.5_dp,0._dp])<1.e-14_dp),'analytic DET false positives')
  call require(all(abs(b-[0._dp,.5_dp,.5_dp])<1.e-14_dp),'analytic DET false negatives')
  call require(all(thresholds==[.35_dp,.4_dp,.8_dp]),'trimmed DET thresholds')
  call require(abs(auc([1._dp,.5_dp,0._dp],[0._dp,1._dp,1._dp])-.75_dp)<1.e-14_dp,'descending generic AUC')
  call calibration_curve([0,0,1,1,1],[0._dp,.25_dp,.5_dp,.75_dp,1._dp],a,b,4)
  call require(all(a==[0._dp,1._dp,1._dp,1._dp]),'uniform calibration left-edge assignment')
  call require(all(b==[.125_dp,.5_dp,.75_dp,1._dp]),'uniform calibration means')
  call calibration_curve(y,[.1_dp,.1_dp,.1_dp,.9_dp],a,b,2,'quantile')
  call require(maxval(abs(a-[1._dp/3,1._dp]))<1.e-14_dp,'repeated quantile edges')
  call require(maxval(abs(b-[.1_dp,.9_dp]))<1.e-14_dp,'quantile calibration means')
  call require(abs(brier_score_loss([0,1],[.7_dp,.2_dp],sample_weight=[2._dp,3._dp])-.58_dp)<1.e-14_dp, &
    'weighted Brier mean squared probability error')
  call require(abs(brier_score_loss([7,7],[.1_dp,.3_dp])-.65_dp)<1.e-14_dp,'Brier singleton maximum-label fallback')
  call require(abs(brier_score_loss([-1,-1],[.1_dp,.3_dp])-.05_dp)<1.e-14_dp,'Brier canonical negative singleton')
  call require(average_precision_score([7,7],[.1_dp,.3_dp])==0,'AP singleton positive-label default')
  call require(average_precision_score([7,7],[.1_dp,.3_dp],7)==1,'explicit singleton positive AP')
  call roc_curve([1,1],[.1_dp,.3_dp],a,b,thresholds)
  call require(all(ieee_is_nan(a)).and.all(ieee_is_finite(b)),'undefined ROC false positive rates')
  call require(ieee_is_nan(roc_auc_score([1,1],[.1_dp,.3_dp])),'single-class ROC AUC is NaN')
  call require(ieee_is_nan(roc_auc_score(y,scores,sample_weight=[0._dp,0._dp,1._dp,2._dp])), &
    'zero-weight negative class gives NaN full ROC AUC')
  call require(abs(average_precision_score([0,1,0,1],[.1_dp,.2_dp,.3_dp,.4_dp], &
    sample_weight=[-1._dp,2._dp,3._dp,4._dp])-8._dp/9)<1.e-14_dp,'source signed-weight AP')
  allocate(large_y(100000),large_scores(100000))
  do i=1,100000
    large_y(i)=mod(i,2); large_scores(i)=real(i,dp)
  end do
  call require(abs(roc_auc_score(large_y,large_scores)-.49999_dp)<1.e-12_dp,'large stable-sort rank identity')
  print *, 'ranking and calibration: PASS'
end program
