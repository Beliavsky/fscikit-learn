program test_group_splitters
  use sklearn, only: group_kfold,leave_one_out,leave_one_group_out,predefined_split,index_split
  use sklearn_common, only: require
  implicit none
  type(group_kfold) :: g
  type(leave_one_out) :: loo
  type(leave_one_group_out) :: logo
  type(predefined_split) :: ps
  type(index_split), allocatable :: folds(:)
  integer :: groups(10)=[9,2,9,2,9,7,9,7,9,3],seen(10),i,j
  g%n_splits=3; folds=g%split(groups); seen=0
  do i=1,size(folds)
    seen(folds(i)%test)=seen(folds(i)%test)+1
    do j=1,size(folds(i)%test)
      call require(.not.any(groups(folds(i)%train)==groups(folds(i)%test(j))),'no group leakage')
    end do
  end do
  call require(all(seen==1),'group test folds exhaustive')
  g%n_splits=4; folds=g%split(groups)
  call require(all(groups(folds(1)%test)==9),'largest group assigned first')
  folds=loo%split(10)
  do i=1,10
    call require(all(folds(i)%test==i).and.size(folds(i)%train)==9,'leave one out')
    call require(.not.any(folds(i)%train==i),'held out sample excluded')
  end do
  folds=logo%split(groups)
  call require(size(folds)==4.and.all(folds(1)%test==[2,4]),'groups sorted by label')
  folds=ps%split([-1,3,3,-2,7,-1])
  call require(size(folds)==3.and.all(folds(1)%test==[4]),'negative fold labels')
  do i=1,3
    call require(any(folds(i)%train==1).and.any(folds(i)%train==6),'training-only samples')
  end do
  folds=ps%split([-1,-1]); call require(size(folds)==0,'all training-only')
  folds=ps%split([integer::]); call require(size(folds)==0,'empty predefined folds')
  folds=ps%split([5,5]); call require(size(folds(1)%train)==0,'empty training fold allowed')
  print *, 'group splitters tests passed'
end program
