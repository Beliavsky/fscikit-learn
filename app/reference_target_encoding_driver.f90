program reference_target_encoding_driver
  use sklearn, only: dp,target_encoder
  implicit none
  type(target_encoder) :: encoder
  real(dp), allocatable :: x(:,:),y(:),query(:,:),z(:,:),q(:,:)
  integer, allocatable :: assignment(:)
  integer :: input,output,n,p,m,i,j,mode,automatic,shuffled,specified,ncat
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) n,p,m,mode
  read(input,*) encoder%target_type,automatic,encoder%smooth,encoder%cv,shuffled,encoder%random_state,specified
  encoder%auto_smooth=automatic==1; encoder%shuffle=shuffled==1
  allocate(x(n,p),y(n),query(m,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  read(input,*) y
  do i=1,m
    read(input,*) query(i,:)
  end do
  if(specified==1) then
    allocate(encoder%categories(p))
    do j=1,p
      read(input,*) ncat
      allocate(encoder%categories(j)%values(ncat)); read(input,*) encoder%categories(j)%values
    end do
  end if
  if(mode==1) then
    z=encoder%fit_transform(x,y); call emit('cross',reshape(z,[size(z)]))
    allocate(assignment(n))
    do j=1,encoder%cv
      assignment(encoder%cv_splits_(j)%test)=j
    end do
    call emit('fold',real(assignment,dp))
  else
    call encoder%fit(x,y)
  end if
  z=encoder%transform(x); q=encoder%transform(query)
  call emit('train',reshape(z,[size(z)])); call emit('query',reshape(q,[size(q)]))
  call emit('shape',real(shape(q),dp)); call emit('mean',encoder%target_mean_)
  call emit('type',[real(merge(0,merge(1,2,encoder%target_type_=='binary'),encoder%target_type_=='continuous'),dp)])
  if(allocated(encoder%classes_)) then
    call emit('classes',encoder%classes_)
  else
    call emit('classes',[real(dp)::])
  end if
  call emit('counts',[(real(size(encoder%categories_(j)%values),dp),j=1,p)])
  write(output,'(a)',advance='no') 'categories'
  do j=1,p
    write(output,'(*(1x,es25.16e3))',advance='no') encoder%categories_(j)%values
  end do
  write(output,*)
  write(output,'(a)',advance='no') 'encodings'
  do j=1,size(encoder%encodings_)
    write(output,'(*(1x,es25.16e3))',advance='no') encoder%encodings_(j)%values
  end do
  write(output,*)
  ! Hyperparameters are consumed at fit time; transform must use learned state.
  encoder%target_type='binary'; encoder%smooth=19; encoder%cv=2
  q=encoder%transform(query); call emit('snapshot',reshape(q,[size(q)]))
  ! Refit to a new feature width, target kind and vocabulary.
  if(allocated(encoder%categories)) deallocate(encoder%categories)
  encoder%target_type='continuous'; call encoder%fit(reshape([1._dp,1._dp,2._dp],[3,1]),[1._dp,3._dp,8._dp])
  q=encoder%transform(reshape([1._dp,2._dp,3._dp],[3,1])); call emit('refit',reshape(q,[size(q)]))
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
