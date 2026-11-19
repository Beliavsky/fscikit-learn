program test_knn_impute
  use sklearn, only: dp,knn_imputer,missing_indicator
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(knn_imputer) :: model
  type(missing_indicator) :: indicator
  real(dp) :: x(3,2),query(1,2),empty(2,2),extended(3,3),q(1,3),nan
  real(dp), allocatable :: result(:,:),again(:,:)
  logical, allocatable :: mask(:,:)
  nan=ieee_value(0._dp,ieee_quiet_nan)
  x(1,:)=[0._dp,10._dp]; x(2,:)=[2._dp,20._dp]; x(3,:)=[4._dp,40._dp]
  query(1,:)=[1._dp,nan]; model%n_neighbors=2; call model%fit(x); result=model%transform(query)
  call require(abs(result(1,2)-15._dp)<1.e-14_dp,'uniform nearest-donor average')
  model%weights='distance'; model%n_neighbors=3; result=model%transform(query)
  call require(abs(result(1,2)-130._dp/7)<1.e-13_dp,'inverse-distance analytic mean')
  query(1,1)=0; result=model%transform(query)
  call require(result(1,2)==10,'zero-distance donor suppresses nonzero distances')
  query=nan; result=model%transform(query)
  call require(maxval(abs(result(1,:)-[2._dp,70._dp/3]))<1.e-13_dp,'no overlap falls back to training means')
  extended(:,:2)=x; extended(:,3)=nan
  model%add_indicator=.true.; call model%fit(extended); q(1,:)=[1._dp,nan,7._dp]
  result=model%transform(q)
  call require(size(result,2)==3.and.result(1,3)==0,'drop empty feature and append original missing indicator')
  model%keep_empty_features=.true.; result=model%transform(q)
  call require(size(result,2)==4.and.result(1,3)==0,'retained empty features always become zero')
  call require(ieee_is_nan(q(1,2)).and.q(1,3)==7,'transform preserves original receiver')
  empty=nan; model%keep_empty_features=.false.; model%add_indicator=.false.
  result=model%fit_transform(empty)
  call require(size(result,2)==0,'all-empty fit can produce zero output columns')
  model%keep_empty_features=.true.; result=model%transform(empty)
  call require(all(result==0),'all-empty retained features become zero')
  extended(1,2)=nan; mask=indicator%fit_transform(extended)
  call require(all(indicator%features_==[2,3]).and.size(mask,2)==2,'fit selects columns with missing values')
  indicator%error_on_new=.false.; q=nan; mask=indicator%transform(q)
  call require(all(mask).and.size(mask,2)==2,'new missing features do not change indicator width')
  indicator%features='all'; mask=indicator%fit_transform(x)
  call require(size(mask,2)==2.and..not.any(mask),'all-feature indicator retains complete columns')
  call model%fit(x); result=model%transform(x); again=model%transform(x)
  call require(all(result==x).and.all(result==again),'complete rows are unchanged and repeated transform is stable')
  print *, 'KNN imputation and missing indicator tests passed'
end program
