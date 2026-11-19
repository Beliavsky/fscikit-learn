program test_ridge_classification
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(ridge_classifier) :: model,refit
  type(ridge_classifier_cv) :: cv
  real(dp) :: pair(2,1),data(8,2),w(8),wide(4,6),error,target
  real(dp), allocatable :: score(:,:),first(:,:),stored(:,:,:)
  integer :: y(8),i,j,keep(7),counter
  pair(:,1)=[-1._dp,1._dp]; model%alpha=2
  call model%fit(pair,[-3,9])
  call require(abs(model%coef_(1,1)-.5_dp)<1.e-14_dp.and.abs(model%intercept_(1))<1.e-14_dp,'analytic ridge slope')
  call model%fit(pair,[-3,9],sample_weight=[1._dp,3._dp])
  call require(abs(model%coef_(1,1)-.6_dp)<1.e-14_dp.and.abs(model%intercept_(1)-.2_dp)<1.e-14_dp, &
    'weighted centering and ridge solution')
  first=model%coef_; call model%fit(pair,[-3,9],class_weight=[1._dp,3._dp])
  call require(maxval(abs(first-model%coef_))<1.e-14_dp,'explicit class weights multiply row weights')
  do i=1,8
    data(i,:)=[real(i-4,dp),sin(real(i,dp))]; w(i)=real(mod(i,3)+1,dp)
  end do
  y=[-3,-3,9,-3,9,9,-3,9]; w(1)=0
  cv%alphas=[.1_dp,1._dp,10._dp]; cv%store_cv_results=.true.
  call cv%fit(data,y,sample_weight=w); stored=cv%cv_results_
  do j=1,3
    refit%alpha=cv%alphas(j)
    do i=1,8
      counter=0
      block
        integer :: k
        do k=1,8
          if(k==i) cycle
          counter=counter+1; keep(counter)=k
        end do
      end block
      call refit%fit(data(keep,:),y(keep),sample_weight=w(keep))
      score=refit%decision_function(data(i:i,:)); target=merge(1._dp,-1._dp,y(i)==9)
      error=w(i)*(target-score(1,1))**2
      call require(abs(error-stored(i,1,j))<1.e-9_dp,'weighted analytic LOO matches explicit refits')
    end do
    call require(abs(cv%cv_scores_(j)+sum(stored(:,:,j))/8)<1.e-12_dp,'CV mean squared error convention')
  end do
  refit%alpha=cv%alpha_; call refit%fit(data,y,sample_weight=w)
  call require(maxval(abs(refit%coef_-cv%coef_))<1.e-12_dp,'selected alpha refits complete data')
  call require(cv%best_score_==maxval(cv%cv_scores_),'CV maximizes negative MSE')
  wide=0
  do i=1,4
    wide(i,i)=1
  end do
  wide(:,5)=2; wide(:,6)=-wide(:,1)
  model%alpha=0; call model%fit(wide,[-3,-3,9,9]); score=model%decision_function(wide)
  call require(model%rank_==3.and.maxval(abs(score(:,1)-[-1._dp,-1._dp,1._dp,1._dp]))<1.e-12_dp, &
    'rank-deficient wide minimum-norm solution')
  cv%alphas=[1.e-5_dp,.1_dp,10._dp]; call cv%fit(wide,[-3,-3,9,9])
  call require(cv%rank_==3.and.all(ieee_is_finite(cv%cv_results_)),'stable full-row-span LOO calculation')
  cv%store_cv_results=.false.; call cv%fit(data,y)
  call require(.not.allocated(cv%cv_results_),'refit clears optional CV results')
  data=0; model%alpha=1; call model%fit(data,y); score=model%decision_function(data)
  call require(all(model%predict(data)==-3).and.all(score==0),'binary zero decisions select lower label')
  print *, 'Ridge classification tests passed'
end program
