program distance_kernel_measures
  use sklearn, only: dp,knn_imputer,haversine_distances,nan_euclidean_distances,cosine_distances, &
    laplacian_kernel,additive_chi2_kernel,chi2_kernel,paired_euclidean_distances
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(knn_imputer) :: imputer
  real(dp) :: locations(2,2),incomplete(2,3),histogram(2,3)
  real(dp), allocatable :: distance(:,:),filled(:,:),kernel(:,:),paired(:)
  locations(1,:)=[0._dp,0._dp];locations(2,:)=[0._dp,acos(-1._dp)/2]
  distance=haversine_distances(locations)
  print *, 'Quarter-equator angle:',distance(1,2)
  incomplete(1,:)=[1._dp,ieee_value(0._dp,ieee_quiet_nan),3._dp]
  incomplete(2,:)=[4._dp,8._dp,3._dp]
  distance=nan_euclidean_distances(incomplete);filled=imputer%fit_transform(incomplete)
  print *, 'Missing-aware distance:',distance(1,2)
  print *, 'Imputed coordinate:',filled(1,2)
  histogram(1,:)=[1._dp,2._dp,0._dp];histogram(2,:)=[0._dp,1._dp,3._dp]
  kernel=chi2_kernel(histogram);print *, 'Chi-squared self similarity:',kernel(1,1),kernel(2,2)
  kernel=laplacian_kernel(histogram);print *, 'Laplacian cross similarity:',kernel(1,2)
  kernel=additive_chi2_kernel(histogram);print *, 'Additive chi-squared cross similarity:',kernel(1,2)
  distance=cosine_distances(histogram);print *, 'Cosine self distances:',distance(1,1),distance(2,2)
  paired=paired_euclidean_distances(histogram,histogram);print *, 'Paired identity distances:',paired
end program
