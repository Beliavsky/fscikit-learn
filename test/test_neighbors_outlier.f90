program test_neighbors_outlier
  use sklearn_common
  use sklearn_neighbors
  use sklearn_outlier
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(radius_neighbors_classifier) :: rc
  type(radius_neighbors_regressor) :: rr
  type(nearest_centroid) :: nc
  type(local_outlier_factor) :: lof
  real(dp) :: x(6,2),targets(6),query(1,2),values(1)
  real(dp), allocatable :: prob(:,:)
  integer :: y(6),labels(6)
  x(:,1)=[-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp]; x(:,2)=x(:,1)
  y=[-7,-7,-7,4,4,4]; targets=x(:,1)**2
  rc%radius=1.5_dp; call rc%fit(x,y)
  call require(rc%score(x,y)==1,'radius classification')
  prob=rc%predict_proba(x); call require(all(abs(sum(prob,dim=2)-1)<1.e-12_dp),'radius probabilities')
  query=100; rc%allow_outliers=.true.; rc%outlier_label=-99
  call require(all(rc%predict(query)==-99),'explicit outlier label')
  rc%outlier_label=4; call require(all(rc%predict(query)==4),'known outlier label')
  rr%radius=1.5_dp; rr%weights='distance'; call rr%fit(x,targets)
  call require(maxval(abs(rr%predict(x)-targets))<1.e-12_dp,'radius exact-match regression')
  values=rr%predict(query); call require(ieee_is_nan(values(1)),'empty radius regression returns NaN')
  call nc%fit(x,y); call require(nc%score(x,y)==1,'nearest centroid classification')
  nc%metric='manhattan'; call nc%fit(x,y); call require(nc%score(x,y)==1,'median centroid')
  nc%shrink_threshold=100; call nc%fit(x,y)
  call require(all(abs(nc%centroids_)<1.e-12_dp),'centroid shrinkage to global mean')
  x(:,1)=[0._dp,0.1_dp,0.2_dp,0.3_dp,0.4_dp,10._dp]; x(:,2)=x(:,1)*2
  lof%n_neighbors=2; labels=lof%fit_predict(x)
  call require(labels(6)==-1.and.all(labels(:5)==1),'LOF isolated outlier')
  lof%novelty=.true.; call lof%fit(x(:5,:)); query=20
  call require(all(lof%predict(query)==-1),'LOF novelty query')
  lof%novelty=.false.; lof%contamination=0.2_dp; labels=lof%fit_predict(x)
  call require(lof%offset_<0,'LOF percentile threshold')
  x=0; lof%n_neighbors=20; call lof%fit(x)
  call require(lof%n_neighbors_==5,'LOF neighbor clamping')
  call require(maxval(abs(lof%negative_outlier_factor_+1))<1.e-12_dp,'LOF duplicate stability')
  print *, 'radius neighbors, centroids and LOF: PASS'
end program
