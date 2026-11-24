program test_pairwise_measures
  use sklearn_common, only: dp,require
  use sklearn_metrics, only: cosine_distances,nan_euclidean_distances,haversine_distances, &
    paired_euclidean_distances,paired_manhattan_distances,paired_cosine_distances,paired_distances, &
    laplacian_kernel,additive_chi2_kernel,chi2_kernel
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  real(dp), allocatable :: x(:,:),y(:,:),d(:,:),v(:)
  type(csr_matrix) :: sx,sy
  real(dp) :: nan,pi
  nan=ieee_value(0._dp,ieee_quiet_nan);pi=acos(-1._dp)
  x=reshape([0._dp,1._dp,0._dp,0._dp],[2,2]);y=x
  d=cosine_distances(x)
  call require(d(1,1)==0.and.d(1,2)==1.and.d(2,2)==0,'cosine zero self diagonal and cross distances')
  d=cosine_distances(x,y);call require(d(1,1)==1,'explicit zero cross vectors have distance one')
  y(1,:)=[1._dp,0._dp]
  v=paired_cosine_distances(x,y);call require(v(1)==.5_dp,'paired cosine zero/unit is one half')
  v=paired_distances(x,y,'cityblock');call require(v(1)==1,'paired named alias')
  v=paired_distances(x,y,custom_distance);call require(v(1)==1,'paired row callback')
  sx=csr_from_dense(x);sy=csr_from_dense(y)
  call require(all(paired_euclidean_distances(sx,sy)==paired_euclidean_distances(x,y)),'CSR paired Euclidean')
  call require(all(paired_manhattan_distances(sx,y)==paired_manhattan_distances(x,y)),'mixed paired Manhattan')
  call require(maxval(abs(cosine_distances(sx,sy)-cosine_distances(x,y)))<1.e-14_dp,'CSR cosine')
  call require(maxval(abs(laplacian_kernel(sx,sy)-laplacian_kernel(x,y)))<1.e-14_dp,'CSR Laplacian')
  x=reshape([1.e-16_dp],[1,1]);y=reshape([1._dp],[1,1]);sx=csr_from_dense(x);sy=csr_from_dense(y)
  v=paired_cosine_distances(sx,sy);call require(v(1)==0,'sparse tiny row normalizes to unit')
  v=paired_cosine_distances(x,y);call require(abs(v(1)-.5_dp)<1.e-14_dp,'dense tiny row follows zero-scale threshold')
  x=reshape([1._dp,nan,nan,nan,3._dp,nan],[2,3]);y=reshape([4._dp,8._dp,3._dp],[1,3])
  d=nan_euclidean_distances(x,y,squared=.true.)
  call require(abs(d(1,1)-13.5_dp)<1.e-14_dp.and.ieee_is_nan(d(2,1)),'missing coordinate weights and no overlap')
  call require(ieee_is_nan(x(1,2)),'missing input remains unchanged')
  d=nan_euclidean_distances(x);call require(d(1,1)==0.and.ieee_is_nan(d(2,2)),'missing self diagonal')
  x=reshape([0._dp,1._dp],[1,2]);y=reshape([0._dp,3._dp],[1,2])
  d=additive_chi2_kernel(x,y);call require(d(1,1)==-1,'additive chi squared and zero denominator')
  d=chi2_kernel(x,y);call require(abs(d(1,1)-exp(-1._dp))<1.e-14_dp,'exponential chi squared')
  d=laplacian_kernel(x,y);call require(abs(d(1,1)-exp(-1._dp))<1.e-14_dp,'default Laplacian bandwidth')
  x=reshape([0._dp,0._dp],[1,2]);y=reshape([0._dp,pi/2],[1,2])
  d=haversine_distances(x,y);call require(abs(d(1,1)-pi/2)<1.e-14_dp,'haversine radians')
  sx=csr_from_dense(x);sy=csr_from_dense(y)
  call require(maxval(abs(haversine_distances(sx,sy)-d))<1.e-14_dp,'CSR haversine')
  print *, 'NaN/paired/angular distances, sparse normalization and histogram kernels: PASS'
contains
  real(dp) function custom_distance(a,b) result(value)
    real(dp), intent(in) :: a(:),b(:)
    value=maxval(abs(a-b))
  end function
end program
