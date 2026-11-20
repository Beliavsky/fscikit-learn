program reference_group_splitters_driver
  use sklearn, only: group_kfold,leave_one_out,leave_one_group_out,predefined_split,index_split
  implicit none
  type(group_kfold) :: g
  type(leave_one_out) :: loo
  type(leave_one_group_out) :: logo
  type(predefined_split) :: ps
  integer, allocatable :: groups(:),assignment(:)
  integer :: input,output,n,k
  character(512) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,k; allocate(groups(n),assignment(n))
  read(input,*) groups; read(input,*) assignment; close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  g%n_splits=k
  call emit('group',g%split(groups)); call emit('loo',loo%split(n))
  call emit('logo',logo%split(groups)); call emit('predefined',ps%split(assignment))
  call emit('all_train',ps%split([-1,-1])); call emit('empty',ps%split([integer::]))
  call emit('all_test',ps%split([8,8]))
  close(output)
contains
  subroutine emit(name,folds)
    character(*), intent(in) :: name
    type(index_split), intent(in) :: folds(:)
    integer :: i
    character(32) :: prefix
    write(output,'(a,1x,i0)') name//'_count',size(folds)
    do i=1,size(folds)
      write(prefix,'(a,i0)') trim(name)//'_',i
      write(output,'(a,*(1x,i0))') trim(prefix)//'_train',folds(i)%train
      write(output,'(a,*(1x,i0))') trim(prefix)//'_test',folds(i)%test
    end do
  end subroutine
end program
