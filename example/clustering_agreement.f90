program clustering_agreement
  use sklearn, only: dp,normalized_mutual_info_score,adjusted_mutual_info_score, &
    fowlkes_mallows_score,contingency_matrix,pair_confusion_matrix,entropy, &
    expected_mutual_information,homogeneity_completeness_v_measure
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: truth(8)=[0,0,0,0,1,1,1,1],predicted(8)=[4,4,4,9,9,9,9,4],i
  integer(int64), allocatable :: counts(:,:)
  integer(int64) :: pairs(2,2)
  real(dp) :: h,c,v
  counts=contingency_matrix(truth,predicted); pairs=pair_confusion_matrix(truth,predicted)
  print *, 'Contingency rows follow sorted true labels; columns follow sorted predicted labels:'
  do i=1,size(counts,1)
    print *, counts(i,:)
  end do
  print *, 'Ordered pair confusion rows: TN FP / FN TP'
  print *, pairs(1,:)
  print *, pairs(2,:)
  print *, 'Normalized mutual information:',normalized_mutual_info_score(truth,predicted)
  print *, 'Adjusted mutual information:',adjusted_mutual_info_score(truth,predicted)
  print *, 'Fowlkes-Mallows score:',fowlkes_mallows_score(truth,predicted)
  print *, 'True-label entropy and expected MI:',entropy(truth),expected_mutual_information(counts,size(truth))
  call homogeneity_completeness_v_measure(truth,predicted,h,c,v,beta=2._dp)
  print *, 'Homogeneity, completeness and beta=2 V-measure:',h,c,v
end program
