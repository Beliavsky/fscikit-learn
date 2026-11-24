program classification_rank_scores
  use sklearn, only: dp,make_blobs,logistic_regression,hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
  implicit none
  type(logistic_regression) :: model
  real(dp), allocatable :: x(:,:),decisions(:,:),probability(:,:)
  integer, allocatable :: y(:)
  real(dp) :: relevance(1,5),ranking(1,5)
  call make_blobs(90,2,3,x,y,random_state=42)
  model%max_iter=500;model%tol=1.e-8_dp;call model%fit(x,y)
  decisions=model%decision_function(x);probability=model%predict_proba(x)
  print *, 'Class score shape:',shape(decisions)
  print *, 'Multiclass hinge loss:',hinge_loss(y,decisions,labels=model%classes_)
  print *, 'Top-two accuracy:',top_k_accuracy_score(y,probability,k=2,labels=model%classes_)
  relevance(1,:)=[10._dp,0._dp,0._dp,1._dp,5._dp]
  ranking(1,:)=[1._dp,0._dp,0._dp,0._dp,1._dp]
  print *, 'Tie-averaged DCG at one:',dcg_score(relevance,ranking,k=1)
  print *, 'Tie-averaged NDCG at one:',ndcg_score(relevance,ranking,k=1)
  print *, 'Ideal NDCG:',ndcg_score(relevance,relevance)
end program
