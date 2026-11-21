program test_resampling
  use sklearn, only: dp,shuffle_split,stratified_shuffle_split,group_shuffle_split,repeated_kfold, &
    repeated_stratified_kfold,leave_p_out,leave_p_groups_out,index_split
  use sklearn_common, only: require
  implicit none
  type(shuffle_split) :: ss
  type(stratified_shuffle_split) :: st
  type(group_shuffle_split) :: gs
  type(repeated_kfold) :: rk
  type(repeated_stratified_kfold) :: rs
  type(leave_p_out) :: lp
  type(leave_p_groups_out) :: lg
  type(index_split), allocatable :: folds(:),again(:)
  integer :: i,j,r,k,seen(12),y(12)=[9,9,9,9,-2,-2,-2,-2,3,3,3,3]
  integer :: groups(12)=[9,9,9,9,9,-3,-3,-3,2,2,7,7]
  call require(ss%get_n_splits()==10.and.st%get_n_splits()==10,'shuffle defaults')
  call require(rk%get_n_splits()==50.and.rs%get_n_splits()==50,'repeated defaults')
  ss%n_splits=5; ss%random_state=42; folds=ss%split(12); again=ss%split(12)
  call require(size(folds)==ss%get_n_splits(),'shuffle fold count')
  do i=1,5
    call require(size(folds(i)%test)==2.and.size(folds(i)%train)==10,'shuffle default ceil/complement')
    call require(all(folds(i)%test==again(i)%test),'shuffle deterministic local RNG')
    call no_overlap(folds(i))
  end do
  ss%test_size=.25_dp; ss%train_size=.5_dp; folds=ss%split(12)
  call require(size(folds(1)%train)==6.and.size(folds(1)%test)==3,'partial shuffle leaves samples unused')
  ss%test_size=-1; ss%train_size=-1; ss%test_count=2; ss%train_count=4; folds=ss%split(12)
  call require(size(folds(1)%train)==4.and.size(folds(1)%test)==2,'integer split sizes')
  st%test_size=.5_dp; st%n_splits=4; folds=st%split(y)
  do i=1,size(folds)
    call no_overlap(folds(i))
    do j=1,3
      k=y(4*j)
      call require(count(y(folds(i)%test)==k)==2.and.count(y(folds(i)%train)==k)==2,'stratification quotas')
    end do
  end do
  call require(gs%get_n_splits()==5,'group shuffle defaults to five iterations')
  call require(ss%get_n_splits()==5.and.st%get_n_splits()==4,'configured shuffle iteration counts')
  folds=gs%split(groups)
  call require(size(folds)==gs%get_n_splits(),'group shuffle count')
  do i=1,size(folds)
    call no_overlap(folds(i))
    do j=1,size(folds(i)%test)
      call require(.not.any(groups(folds(i)%train)==groups(folds(i)%test(j))),'whole group holdout')
    end do
  end do
  rk%n_splits=5; rk%n_repeats=3; folds=rk%split(12)
  call require(size(folds)==rk%get_n_splits(),'repeated fold count')
  do r=1,3
    seen=0
    do k=1,5
      i=(r-1)*5+k; seen(folds(i)%test)=seen(folds(i)%test)+1; call no_overlap(folds(i))
      call require(size(folds(i)%test)==2+merge(1,0,k<=2),'uneven repeated fold sizes')
      call require(all(folds(i)%test(2:)>folds(i)%test(:size(folds(i)%test)-1)),'sorted repeated tests')
    end do
    call require(all(seen==1),'each repeated sample held out once per repeat')
  end do
  rs%n_splits=3; rs%n_repeats=2; folds=rs%split(y)
  call require(size(folds)==rs%get_n_splits(),'repeated stratified count')
  do r=1,2
    seen=0
    do k=1,3
      i=(r-1)*3+k; seen(folds(i)%test)=seen(folds(i)%test)+1; call no_overlap(folds(i))
      do j=1,3
        call require(count(y(folds(i)%test)==y(4*j))>=1,'repeated class balance')
      end do
    end do
    call require(all(seen==1),'repeated stratified complete partition')
  end do
  rs%n_splits=3; folds=rs%split([4,9,9,9,9])
  call require(size(folds)==6,'rare classes accepted when another class fills all folds')
  lp%p=2; folds=lp%split(4)
  call require(size(folds)==6.and.lp%get_n_splits(4)==6,'binomial leave-p count')
  call require(all(folds(1)%test==[1,2]).and.all(folds(2)%test==[1,3]),'lexicographic combinations')
  call require(all(folds(6)%test==[3,4]),'last combination')
  do i=1,6
    call no_overlap(folds(i))
  end do
  lp%p=3; folds=lp%split(4)
  call require(size(folds)==4.and.size(folds(1)%train)==1,'leave all but one')
  lg%n_groups=2; folds=lg%split(groups)
  call require(size(folds)==6.and.lg%get_n_splits(groups)==6,'leave-p groups count')
  call require(all(folds(1)%test==[6,7,8,9,10]),'leave group combinations sorted by label')
  do i=1,6
    do j=1,size(folds(i)%test)
      call require(.not.any(groups(folds(i)%train)==groups(folds(i)%test(j))),'leave-p groups no leakage')
    end do
  end do
  print *, 'resampling splitters: PASS'
contains
  subroutine no_overlap(fold)
    type(index_split), intent(in) :: fold
    integer :: j
    do j=1,size(fold%test)
      call require(.not.any(fold%train==fold%test(j)),'split sample leakage')
    end do
  end subroutine
end program
