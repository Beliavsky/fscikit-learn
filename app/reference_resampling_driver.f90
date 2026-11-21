program reference_resampling_driver
  use sklearn, only: dp,index_split,shuffle_split,stratified_shuffle_split,group_shuffle_split, &
    repeated_kfold,repeated_stratified_kfold,leave_p_out,leave_p_groups_out
  implicit none
  type(shuffle_split) :: ss
  type(stratified_shuffle_split) :: st
  type(group_shuffle_split) :: gs
  type(repeated_kfold) :: rk
  type(repeated_stratified_kfold) :: rs
  type(leave_p_out) :: lp
  type(leave_p_groups_out) :: lg
  type(index_split), allocatable :: folds(:),again(:)
  integer, allocatable :: y(:),groups(:)
  integer :: method,n,splits,repeats,seed,tc,trc,p,unit,out,i,declared
  real(dp) :: te,tr
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) method,n,splits,repeats,seed,te,tr,tc,trc,p
  allocate(y(n),groups(n)); read(unit,*) y; read(unit,*) groups; close(unit)
  ss%n_splits=splits; ss%random_state=seed; ss%test_size=te; ss%train_size=tr
  ss%test_count=tc; ss%train_count=trc
  st%n_splits=splits; st%random_state=seed; st%test_size=te; st%train_size=tr
  st%test_count=tc; st%train_count=trc
  gs%n_splits=splits; gs%random_state=seed; gs%test_size=te; gs%train_size=tr
  gs%test_count=tc; gs%train_count=trc
  rk%n_splits=splits; rk%n_repeats=repeats; rk%random_state=seed
  rs%n_splits=splits; rs%n_repeats=repeats; rs%random_state=seed
  lp%p=p; lg%n_groups=p
  select case(method)
  case(1)
    folds=ss%split(n); again=ss%split(n); declared=ss%get_n_splits()
  case(2)
    folds=st%split(y); again=st%split(y); declared=st%get_n_splits()
  case(3)
    folds=gs%split(groups); again=gs%split(groups); declared=gs%get_n_splits()
  case(4)
    folds=rk%split(n); again=rk%split(n); declared=rk%get_n_splits()
  case(5)
    folds=rs%split(y); again=rs%split(y); declared=rs%get_n_splits()
  case(6)
    folds=lp%split(n); again=lp%split(n); declared=lp%get_n_splits(n)
  case(7)
    folds=lg%split(groups); again=lg%split(groups); declared=lg%get_n_splits(groups)
  end select
  open(newunit=out,file=trim(output),status='replace')
  write(out,*) 'count ',size(folds)
  write(out,*) 'declared ',declared
  write(out,*) 'labels ',y
  write(out,*) 'groups ',groups
  do i=1,size(folds)
    write(out,'(a,i0,a,*(i0,1x))') 'fold_',i,'_train ',folds(i)%train
    write(out,'(a,i0,a,*(i0,1x))') 'fold_',i,'_test ',folds(i)%test
    if(any(folds(i)%train/=again(i)%train).or.any(folds(i)%test/=again(i)%test)) error stop 'repeatability'
  end do
  write(out,*) 'repeatable ',1
  close(out)
end program
