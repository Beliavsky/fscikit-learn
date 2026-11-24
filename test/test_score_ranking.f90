program test_score_ranking
  use sklearn_common, only: dp,require
  use sklearn_metrics, only: hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
  implicit none
  real(dp) :: decision(2,3),relevance(1,5),scores(1,5),v
  decision(1,:)=[2._dp,1._dp,0._dp];decision(2,:)=decision(1,:)
  v=hinge_loss([-1,1,1],[-2._dp,3._dp,.5_dp])
  call require(abs(v-1._dp/6)<1.e-14_dp,'binary hinge margins')
  call require(hinge_loss([7,7],[1._dp,1._dp],labels=[-1,7])==2,'single observed hinge class is negative')
  v=hinge_loss([10,20],decision,labels=[30,10,20,20])
  call require(v==1,'multiclass Crammer-Singer with sorted unique labels')
  decision=0
  v=top_k_accuracy_score([10,20],decision,k=2,labels=[10,20,30])
  call require(v==.5_dp,'higher class indices win multiclass ties')
  v=top_k_accuracy_score([10,20],decision,k=3,labels=[10,20,30])
  call require(v==1,'top-k cutoff at class count')
  v=top_k_accuracy_score([0,1],[.5_dp,.5_dp],k=1)
  call require(v==.5_dp,'binary score exactly at probability threshold is negative')
  v=top_k_accuracy_score([0,1],[-.1_dp,.1_dp],k=1)
  call require(v==1,'binary decision threshold is zero')
  v=top_k_accuracy_score([0,1],[-1._dp,1._dp],k=1,normalize=.false.,sample_weight=[-1._dp,1._dp])
  call require(v==0,'raw weighted count permits zero weight sum')
  relevance(1,:)=[10._dp,0._dp,0._dp,1._dp,5._dp];scores(1,:)=[1._dp,0._dp,0._dp,0._dp,1._dp]
  call require(dcg_score(relevance,scores,k=1)==7.5_dp,'average DCG tie straddles cutoff')
  call require(ndcg_score(relevance,scores,k=1)==.75_dp,'normalized tie average')
  call require(dcg_score(relevance,scores,k=1,ignore_ties=.true.)==5,'ignored ties deterministic order')
  call require(dcg_score(relevance,scores,log_base=1._dp)==0,'unit log base gives zero discounts')
  call require(abs(dcg_score(relevance,scores,log_base=.5_dp)+dcg_score(relevance,scores))<1.e-14_dp, &
    'log bases below one invert discount sign')
  call require(abs(ndcg_score(relevance,relevance)-1)<1.e-14_dp,'ideal relevance normalization')
  call require(ndcg_score(relevance*0,scores)==0,'all irrelevant documents')
  call require(dcg_score(int(relevance),scores)==dcg_score(relevance,scores),'integer relevance adapter')
  call require(dcg_score(relevance>0,scores)==dcg_score(merge(1._dp,0._dp,relevance>0),scores),'logical adapter')
  print *, 'Hinge margins, top-k class ties and discounted relevance ranking: PASS'
end program
