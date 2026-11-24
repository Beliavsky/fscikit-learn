program test_classification_metrics
  use sklearn_common, only: dp,require
  use sklearn_metrics, only: precision_score,recall_score,f1_score,jaccard_score,fbeta_score, &
    precision_recall_fscore_support,multilabel_confusion_matrix
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan
  implicit none
  real(dp), allocatable :: precision(:),recall(:),fscore(:),support(:),raw(:),cm(:,:,:),other(:,:,:)
  integer :: yt(2,3),yp(2,3)
  type(csr_matrix) :: sy,sp
  real(dp) :: v,nan,inf
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call precision_recall_fscore_support([0,1,2,2],[0,2,1,2],precision,recall,fscore,support)
  call require(all(precision==[1._dp,0._dp,.5_dp]).and.all(support==[1._dp,1._dp,2._dp]),'raw per-class summary')
  v=f1_score([0,1,2,2],[0,2,1,2],average='weighted')
  call require(v==.5_dp,'weighted class average')
  v=precision_score([0,1],[0,0],average='macro',labels=[0,1,7],zero_division=nan,raw_values=raw)
  call require(v==.5_dp.and.ieee_is_nan(raw(2)).and.ieee_is_nan(raw(3)),'NaN divisions excluded from averaging')
  v=f1_score([0,1],[0,0],average='macro',labels=[0,1,7],zero_division=nan)
  call require(abs(v-1._dp/3)<1.e-14_dp,'NaN-aware macro F1')
  yt(1,:)=[1,0,1];yt(2,:)=[0,1,0];yp(1,:)=[1,1,0];yp(2,:)=[0,1,1]
  sy=csr_from_dense(real(yt,dp));sp=csr_from_dense(real(yp,dp))
  call require(abs(f1_score(yt,yp,average='micro')-4._dp/7)<1.e-14_dp,'multilabel micro F1')
  call require(abs(f1_score(sy,sp,average='samples')-7._dp/12)<1.e-14_dp,'sparse sample average')
  call require(abs(jaccard_score(sy,yp,average='samples')-5._dp/12)<1.e-14_dp,'mixed Jaccard sample average')
  call require(precision_score(yt,sp,average='micro')==.5_dp,'mixed precision')
  call require(f1_score(yt/=0,yp/=0,average='micro')==f1_score(yt,yp,average='micro'),'logical indicators')
  call require(fbeta_score(yt,yp,0._dp,average='micro')==precision_score(yt,yp,average='micro'),'beta zero precision')
  call require(fbeta_score(yt,yp,inf,average='micro')==recall_score(yt,yp,average='micro'),'infinite beta recall')
  cm=multilabel_confusion_matrix(yt,yp);other=multilabel_confusion_matrix(sy,sp)
  call require(all(shape(cm)==[3,2,2]).and.all(cm==other),'dense/CSR confusion tensor')
  call require(all(cm(:,2,2)==[1._dp,1._dp,0._dp]).and.all(cm(:,1,1)==[1._dp,0._dp,0._dp]),'TP/TN tensor axes')
  cm=multilabel_confusion_matrix(sy,sp,labels=[2,0,2],samplewise=.true.)
  call require(all(cm(:,2,1)==[2._dp,0._dp]).and.all(cm(:,1,2)==[0._dp,2._dp]),'duplicate zero-based label selection')
  call precision_recall_fscore_support(yt,yp,precision,recall,fscore,support,average='samples')
  call require(size(fscore)==1.and..not.allocated(support),'averaged support is absent')
  call precision_recall_fscore_support([integer::],[integer::],precision,recall,fscore,support)
  call require(size(precision)==0.and.allocated(support).and.size(support)==0,'empty vector raw summary')
  call require(precision_score([integer::],[integer::])==0,'empty binary default')
  call require(ieee_is_nan(precision_score([integer::],[integer::],average='macro')),'empty macro NaN')
  sy=csr_from_arrays(2,10000,[1,3,4],[1,10000,5000],[1._dp,1._dp,1._dp])
  sp=csr_from_arrays(2,10000,[1,2,4],[1,5000,9999],[1._dp,1._dp,1._dp])
  call require(abs(f1_score(sy,sp,average='micro')-2._dp/3)<1.e-14_dp,'wide sparse confusion accumulation')
  v=precision_score([0,1],[0,1],average='weighted',sample_weight=[nan,1._dp])
  call require(v==1,'NaN-weighted undefined classes are excluded as in source')
  v=jaccard_score([0,1],[0,1],average='weighted',sample_weight=[inf,1._dp])
  call require(ieee_is_nan(v),'Jaccard preserves nonfinite-weight arithmetic')
  print *, 'Weighted classification summaries, F-beta, sparse indicators and zero-division modes: PASS'
end program
