program test_cluster_information
  use sklearn_metrics, only: normalized_mutual_info_score,adjusted_mutual_info_score, &
    fowlkes_mallows_score,contingency_matrix,pair_confusion_matrix,entropy,expected_mutual_information, &
    homogeneity_completeness_v_measure,homogeneity_score,completeness_score,v_measure_score
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: truth(4)=[0,0,1,2],prediction(4)=[0,0,1,1],i
  integer, allocatable :: large(:)
  integer(int64), allocatable :: table(:,:)
  integer(int64) :: pairs(2,2)
  real(dp), allocatable :: smooth(:,:)
  real(dp) :: h,c,v,expected
  table=contingency_matrix(truth,prediction)
  call require(all(shape(table)==[3,2]),'Cluster contingency dimensions')
  call require(all(table(:,1)==[2_int64,0_int64,0_int64]).and.all(table(:,2)==[0_int64,1_int64,1_int64]), &
    'Cluster contingency sorted label order')
  smooth=contingency_matrix(truth,prediction,eps=.25_dp)
  call require(all(smooth==real(table,dp)+.25_dp),'Cluster additive contingency smoothing')
  pairs=pair_confusion_matrix(truth,prediction)
  call require(all(pairs(1,:)==[8_int64,2_int64]).and.all(pairs(2,:)==[0_int64,2_int64]), &
    'Cluster ordered pair orientation')
  call require(abs(fowlkes_mallows_score(truth,prediction)-1/sqrt(2._dp))<1.e-14_dp,'Fowlkes Mallows analytic pairs')
  call require(abs(entropy([0,0,1,1])-log(2._dp))<1.e-14_dp,'Cluster balanced entropy')
  call require(entropy([integer ::])==1.and.entropy([7,7,7])==0,'Cluster source entropy edge conventions')
  expected=log(2._dp)/3
  call require(abs(expected_mutual_information(reshape([2,0,0,2],[2,2]),4)-expected)<1.e-14_dp, &
    'Expected MI exact balanced hypergeometric expectation')
  call require(abs(expected_mutual_information(reshape([2_int64,0_int64,0_int64,2_int64],[2,2]),4)-expected) &
    <1.e-14_dp,'Expected MI int64 overload')
  call require(abs(normalized_mutual_info_score(truth,truth)-1)<1.e-14_dp,'NMI perfect clustering')
  call require(abs(adjusted_mutual_info_score(truth,truth)-1)<1.e-12_dp,'AMI perfect nondegenerate clustering')
  call require(abs(adjusted_mutual_info_score([0,0,1,1],[0,1,0,1])+.5_dp)<1.e-13_dp, &
    'AMI independent balanced partitions retain negative chance adjustment')
  call require(normalized_mutual_info_score([integer ::],[integer ::])==1,'NMI empty partitions')
  call require(adjusted_mutual_info_score([integer ::],[integer ::])==1,'AMI empty partitions')
  pairs=pair_confusion_matrix([integer ::],[integer ::])
  call require(all(pairs==0),'Pair confusion empty partitions')
  table=contingency_matrix([integer ::],[integer ::])
  call require(all(shape(table)==[0,0]),'Empty contingency dimensions')
  call require(fowlkes_mallows_score([integer ::],[integer ::])==0,'Empty Fowlkes Mallows score')
  call homogeneity_completeness_v_measure([integer ::],[integer ::],h,c,v,beta=0._dp)
  call require(h==1.and.c==1.and.v==1,'Empty HCV source convention')
  call homogeneity_completeness_v_measure([0,0,1,1],[0,1,0,1],h,c,v,beta=0._dp)
  call require(h==0.and.c==0.and.v==0,'Zero-beta independent partitions')
  call homogeneity_completeness_v_measure(truth,prediction,h,c,v,beta=2._dp)
  call require(abs(c-1)<1.e-14_dp.and.abs(v-3*h/(2*h+1))<1.e-14_dp,'Cluster weighted V measure')
  call require(abs(v-v_measure_score(truth,prediction,beta=2._dp))<1.e-14_dp, &
    'Weighted V measure wrapper')
  call require(homogeneity_score([integer ::],[integer ::])==1.and. &
    completeness_score([integer ::],[integer ::])==1.and.v_measure_score([integer ::],[integer ::])==1, &
    'Source-compatible empty wrapper results')
  allocate(large(70000)); large=9
  pairs=pair_confusion_matrix(large,large)
  call require(pairs(2,2)==70000_int64*69999_int64,'Pair counts exceed default integer range safely')
  do i=1,size(large)
    large(i)=mod(i,97)
  end do
  call require(abs(fowlkes_mallows_score(large,-large)-1)<1.e-14_dp,'Large clustering relabel invariance')
  print *, 'Cluster information metrics tests passed'
end program
