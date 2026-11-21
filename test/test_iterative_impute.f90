program test_iterative_impute
  use sklearn_iterative_impute
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_finite
  implicit none
  type(iterative_imputer) :: model,copy,posterior
  real(dp) :: x(8,3),q(2,3),nan
  real(dp), allocatable :: z(:,:),a(:,:),b(:,:)
  character(128), allocatable :: names(:)
  integer :: j
  nan=ieee_value(0._dp,ieee_quiet_nan)
  x(:,1)=[(real(j,dp),j=1,8)]; x(:,2)=2*x(:,1); x(:,3)=3*x(:,1)
  x(2,1)=nan; x(4,2)=nan; x(6,3)=nan
  model%add_indicator=.true.; z=model%fit_transform(x)
  call require(all(ieee_is_finite(z)).and.size(z,2)==6,'chained imputation returns finite values and indicators')
  call require(abs(z(2,1)-2)<.01_dp.and.abs(z(4,2)-8)<.01_dp,'correlated features recover missing values')
  call require(z(2,4)==1.and.z(4,5)==1.and.z(6,6)==1,'indicators preserve original missing mask')
  call require(all(z([1,3,4,5,6,7,8],1)==x([1,3,4,5,6,7,8],1)),'observed values are unchanged')
  a=model%transform(x); call require(maxval(abs(a-z))<.01_dp,'stored regression sequence replays training imputation')
  copy=model; q=x(:2,:); q(1,2)=nan; a=copy%transform(q)
  call require(all(ieee_is_finite(a)),'new missing patterns replay fitted regressions')
  names=copy%get_feature_names_out()
  call require(trim(names(4))=='missingindicator_x0','imputer feature names')
  model%max_iter=0; x(:,3)=nan; z=model%fit_transform(x)
  call require(size(z,2)==5.and.model%n_iter_==0,'empty column is dropped while indicator is retained')
  model%keep_empty_features=.true.; z=model%fit_transform(x)
  call require(size(z,2)==6.and.all(z(:,3)==0),'empty feature retention uses initial fill')
  b=copy%transform(q); call require(all(a==b),'refitting original model preserves independent fitted copy')
  model%initial_strategy='constant'; model%keep_empty_features=.false.; z=model%fit_transform(x)
  names=model%get_feature_names_out()
  call require(size(names)==6.and.size(z,2)==5,'sklearn constant initialization empty-column naming quirk')
  model%initial_strategy='mean'
  model%missing_is_nan=.false.; model%missing_values=-99; x=1; x(1,1)=-99
  z=model%fit_transform(x); call require(z(1,1)==1,'finite missing sentinel')
  posterior%sample_posterior=.true.; posterior%max_iter=3; posterior%random_state=42
  posterior%min_value=0; posterior%max_value=10; x(:,1)=[(real(j,dp),j=1,8)]
  x(:,2)=2*x(:,1); x(:,3)=.5_dp*x(:,1); x(2,1)=nan; x(3,2)=nan
  z=posterior%fit_transform(x); a=posterior%transform(x); b=posterior%transform(x)
  call require(z(2,1)>=0.and.z(2,1)<=10.and.z(3,2)<=10,'posterior respects bounds')
  call require(any(a/=b),'posterior transforms advance persistent random state')
  model=iterative_imputer(); model%skip_complete=.true.; x=2; call model%fit(x)
  call require(model%n_iter_==1.and.size(model%imputation_sequence_)==0,'complete fit needs no chained estimators')
  q=x(:2,:); q(1,2)=nan; a=model%transform(q)
  call require(a(1,2)==2,'skip_complete uses initial statistics for newly missing features')
  print *, 'IterativeImputer: PASS'
end program
