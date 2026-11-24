program test_multilabel_ranking
  use sklearn_metrics, only: coverage_error,label_ranking_average_precision_score,label_ranking_loss
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  implicit none
  integer :: y(2,3),special(2,3)
  real(dp) :: scores(2,3),v,w(2)
  type(csr_matrix) :: csr
  y(1,:)=[1,0,0];y(2,:)=[0,0,1]
  scores(1,:)=[.75_dp,.5_dp,1._dp];scores(2,:)=[1._dp,.2_dp,.1_dp]
  call require(abs(coverage_error(y,scores)-2.5_dp)<1.e-14_dp,'coverage maximal relevant rank')
  call require(abs(label_ranking_average_precision_score(y,scores)-5._dp/12)<1.e-14_dp,'LRAP example')
  call require(abs(label_ranking_loss(y,scores)-.75_dp)<1.e-14_dp,'ranking loss example')
  scores=1
  call require(coverage_error(y,scores)==3,'coverage ties get maximal rank')
  call require(abs(label_ranking_average_precision_score(y,scores)-1._dp/3)<1.e-14_dp,'LRAP ties')
  call require(label_ranking_loss(y,scores)==1,'tied positive/negative pairs are errors')
  special(1,:)=0;special(2,:)=1
  call require(coverage_error(special,scores)==1.5_dp,'empty/full coverage')
  call require(label_ranking_average_precision_score(special,scores)==1,'empty/full precision')
  call require(label_ranking_loss(special,scores)==0,'empty/full loss')
  csr=csr_from_dense(real(special,dp))
  call require(label_ranking_average_precision_score(csr,scores)==1,'CSR empty/full precision')
  call require(label_ranking_loss(csr,scores)==0,'CSR empty/full loss')
  w=[-1._dp,2._dp]
  call require(coverage_error(special,scores,w)==6,'signed sample weights')
  call require(coverage_error(special/=0,scores)==1.5_dp,'logical indicator overload')
  special=-y
  call require(label_ranking_loss(special,scores)==label_ranking_loss(y,scores),'integer two-valued indicators')
  v=label_ranking_average_precision_score(reshape([0,1],[2,1]),reshape([.2_dp,.8_dp],[2,1]))
  call require(v==1,'single-label LRAP degeneracy')
  print *, 'Multilabel ranking, maximal ties, empty/full rows and weighted dense/CSR indicators: PASS'
end program
