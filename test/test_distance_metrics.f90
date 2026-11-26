program test_distance_metrics
 use sklearn,only:dp,distance_metric,get_metric
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 type(distance_metric)::metric
 type(csr_matrix)::sparse,duplicates
 real(dp)::x(3,2),variance(2),precision(2,2),zero(1,2),inf
 real(dp),allocatable::d(:,:)
 logical,allocatable::unknown(:,:)
 integer::calls
 x(1,:)=[1._dp,0._dp];x(2,:)=[0._dp,1._dp];x(3,:)=0;zero=0
 inf=ieee_value(0._dp,ieee_positive_inf)
 metric=get_metric('minkowski',p=1._dp)
 call require(metric%type_name()=='ManhattanDistance64','Minkowski canonical factory')
 metric=get_metric('p',p=-inf)
 call require(metric%type_name()=='ChebyshevDistance64','Source infinite-power canonical factory')
 metric=get_metric('minkowski',p=.5_dp,w=[1._dp,2._dp])
 d=metric%pairwise(x)
 call require(d(1,2)==9,'weighted fractional Minkowski')
 call require(ieee_is_nan(metric%dist_to_rdist(-inf)),'NumPy power half specializes sqrt')
 variance=[1._dp,4._dp];metric=get_metric('seuclidean',v=variance)
 d=metric%pairwise(reshape([3._dp,4._dp],[1,2]),zero)
 call require(d(1,1)==sqrt(13._dp),'SEuclidean variance geometry')
 precision(1,:)=[1._dp,.2_dp];precision(2,:)=[.2_dp,2._dp]
 metric=get_metric('mahalanobis',vi=precision);d=metric%pairwise(x)
 call require(abs(d(1,2)-sqrt(2.6_dp))<1.e-15_dp.and.d(2,2)==0,'dense Mahalanobis')
 sparse=csr_from_dense(x);metric=get_metric('mahalanobis',vi=precision)
 d=metric%pairwise(sparse)
 call require(d(2,2)==1.and.abs(d(3,3)-sqrt(3.4_dp))<1.e-15_dp,'Source CSR work buffer persists')
 d=metric%pairwise(sparse)
 call require(d(1,1)==sqrt(2._dp),'Source CSR metric state persists across calls')
 precision(1,:)=[1._dp,2._dp];precision(2,:)=[3._dp,4._dp]
 metric=get_metric('mahalanobis',v=precision);d=metric%pairwise(sparse)
 call require(abs(d(3,3)-1.0536712127723509e-8_dp)<1.e-22_dp,'LU covariance inverse preserves CSR cancellation')
 precision=0;precision(1,1)=ieee_value(0._dp,ieee_quiet_nan);precision(2,2)=1
 metric=get_metric('mahalanobis',v=precision);d=metric%pairwise(x)
 call require(all(ieee_is_nan(d)),'NaN covariance is accepted and propagated')
 precision=0;precision(1,1)=1;precision(2,2)=1.e-300_dp
 metric=get_metric('mahalanobis',v=precision);d=metric%pairwise(x,zero)
 call require(abs(d(2,1)/1.e150_dp-1)<1.e-14_dp,'small nonzero covariance pivot is accepted')
 duplicates%n_rows=2;duplicates%n_cols=1
 duplicates%indptr=[1,3,3];duplicates%indices=[1,1];duplicates%data=[.25_dp,.75_dp]
 metric=get_metric('euclidean');d=metric%pairwise(duplicates)
 call require(d(1,2)==sqrt(.625_dp),'raw CSR duplicate entries are not merged')
 metric=get_metric('braycurtis');d=metric%pairwise(sparse);unknown=metric%undefined_pairs()
 call require(unknown(1,3).and.ieee_is_nan(d(1,3)),'Source out-of-allocation reads marked as undefined')
 call require(.not.unknown(3,3).and.ieee_is_nan(d(3,3)),'defined zero-denominator CSR Bray NaN')
 calls=0;metric=get_metric(callback);d=metric%pairwise(x)
 call require(calls==6.and.d(1,1)==999._dp,'callback self pairwise includes diagonal')
 d=metric%pairwise(sparse)
 call require(calls==6.and.all(d==-999),'Source CSR callback base-class sentinel')
 print *,'DistanceMetric factories, weighted geometry, raw CSR and Source state passed'
contains
 function callback(u,v)result(value)
  real(dp),intent(in)::u(:),v(:)
  real(dp)::value
  calls=calls+1;value=1000+sum(u)-2*sum(v)
 end function
end program
