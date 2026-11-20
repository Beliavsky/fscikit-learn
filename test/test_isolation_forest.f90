program test_isolation_forest
  use sklearn, only: dp,isolation_forest
  use sklearn_common, only: require
  implicit none
  type(isolation_forest) :: model,direct
  real(dp) :: x(60,2),query(2,2),weights(60)
  real(dp), allocatable :: scores(:),before(:)
  integer, allocatable :: labels(:)
  integer :: i
  do i=1,60
    x(i,:)=[.1_dp*sin(real(i,dp)),.1_dp*cos(real(i,dp))]
  end do
  query(1,:)=[0._dp,0._dp]; query(2,:)=[8._dp,8._dp]
  model%n_estimators=30; model%random_state=42; labels=model%fit_predict(x)
  scores=model%score_samples(query)
  call require(scores(2)<scores(1),'isolation of distant observations')
  call require(all(labels==model%predict(x)),'isolation fit_predict matches predict')
  call require(model%max_samples_==60.and.model%max_features_==2,'isolation automatic counts')
  call require(abs(model%offset_+.5_dp)<1.e-14_dp,'isolation automatic offset')
  before=model%score_samples(x); model%warm_start=.true.; model%n_estimators=40; call model%fit(x)
  direct%n_estimators=40; direct%random_state=42; call direct%fit(x)
  call require(maxval(abs(model%score_samples(x)-direct%score_samples(x)))<1.e-14_dp,'isolation warm-start equivalence')
  model%warm_start=.false.; model%contamination=.2_dp; call model%fit(x)
  scores=model%decision_function(x)
  call require(count(scores<0)==12,'isolation empirical contamination threshold')
  weights=0; model%contamination=-1; call model%fit(x,weights)
  call require(all(model%score_samples(x)==-1),'isolation empty weighted roots')
  model%max_samples=1; call model%fit(x)
  call require(all(model%score_samples(x)==-.5_dp),'isolation singleton normalization')
  model%max_samples=0; model%max_samples_fraction=.001_dp; model%bootstrap=.true.; call model%fit(x)
  call require(model%max_samples_==0.and.all(model%score_samples(x)==-.5_dp),'isolation empty bootstrap convention')
  print *, 'IsolationForest tests passed'
end program
