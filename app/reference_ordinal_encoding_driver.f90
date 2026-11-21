program reference_ordinal_encoding_driver
  use sklearn_ordinal_encoding, only: ordinal_encoder
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(ordinal_encoder) :: encoder
  real(dp), allocatable :: x(:,:),query(:,:),z(:,:),q(:,:),decoded(:,:),codes(:,:)
  logical, allocatable :: unknown(:,:),infrequent(:,:)
  real(dp) :: missing_value,unknown_value
  integer :: input,output,n,p,m,i,j,specified,ncat,has_unknown,inverse_mode,t
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) n,p,m,specified,inverse_mode,t
  read(input,*) encoder%handle_unknown,has_unknown,unknown_value,missing_value, &
    encoder%min_frequency_count,encoder%min_frequency_fraction,encoder%max_categories
  if(has_unknown==1) encoder%unknown_value=unknown_value
  encoder%encoded_missing_value=missing_value
  allocate(x(n,p),query(m,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
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
  if(t>0) then
    allocate(codes(t,p))
    do i=1,t
      read(input,*) codes(i,:)
    end do
  end if
  z=encoder%fit_transform(x); q=encoder%transform(query)
  call emit('train',reshape(z,[size(z)])); call emit('query',reshape(q,[size(q)]))
  call emit('counts',[(real(size(encoder%categories_(j)%values),dp),j=1,p)])
  call emit('rare_counts',[(real(size(encoder%infrequent_categories_(j)%values),dp),j=1,p)])
  write(output,'(a)',advance='no') 'categories'
  do j=1,p
    write(output,'(*(1x,es25.16e3))',advance='no') encoder%categories_(j)%values
  end do
  write(output,*)
  write(output,'(a)',advance='no') 'rare'
  do j=1,p
    write(output,'(*(1x,es25.16e3))',advance='no') encoder%infrequent_categories_(j)%values
  end do
  write(output,*)
  write(output,'(a)',advance='no') 'category_counts'
  do j=1,p
    write(output,'(*(1x,es25.16e3))',advance='no') real(encoder%mappings_(j)%counts,dp)
  end do
  write(output,*)
  call emit('feature_count',[real(encoder%n_features_in_,dp)])
  ! Grouping controls are learned at fit time.
  encoder%max_categories=1; encoder%min_frequency_count=999
  z=encoder%transform(query); call emit('snapshot',reshape(z,[size(z)]))
  if(t==0) codes=q
  if(inverse_mode<0) then
    close(output)
    decoded=encoder%inverse_transform(codes,unknown,infrequent)
    call require(.false.,'source inverse failure was not reproduced')
  end if
  if(inverse_mode>0) then
    decoded=encoder%inverse_transform(codes,unknown,infrequent)
    call emit('inverse',reshape(decoded,[size(decoded)]))
    call emit('unknown',real(reshape(merge(1,0,unknown),[size(unknown)]),dp))
    call emit('infrequent',real(reshape(merge(1,0,infrequent),[size(infrequent)]),dp))
    ! Source inverse mutates its argument; native inverse deliberately preserves it.
    call emit('preserved_codes',reshape(codes,[size(codes)]))
  end if
  if(allocated(encoder%categories)) deallocate(encoder%categories)
  if(allocated(encoder%unknown_value)) deallocate(encoder%unknown_value)
  encoder%encoded_missing_value=ieee_value(0._dp,ieee_quiet_nan); encoder%handle_unknown='error'
  encoder%min_frequency_count=0; encoder%min_frequency_fraction=0; encoder%max_categories=0
  z=encoder%fit_transform(reshape([7._dp,7._dp,ieee_value(0._dp,ieee_quiet_nan)],[3,1]))
  call emit('refit',reshape(z,[size(z)]))
  close(output)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
