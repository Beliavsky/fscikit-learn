program reference_chi2_samplers_driver
  use sklearn
  use sklearn_common, only: seed_rng
  implicit none
  type(additive_chi2_sampler) :: additive
  type(skewed_chi2_sampler) :: skewed
  real(dp), allocatable :: x(:,:),train(:,:),query(:,:),z(:,:),u(:,:),v(:)
  integer :: n,p,i,j,input,output,k
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    additive=additive_chi2_sampler(); additive%sample_steps=j
    select case(j)
    case(4); additive%sample_steps=5; additive%sample_interval=.3_dp
    case(5); additive%sample_steps=2; additive%sample_interval=0
    case(6); additive%sample_steps=4; additive%sample_interval=1.2_dp
    case(7); additive%sample_steps=8; additive%sample_interval=.25_dp
    case(8); additive%sample_steps=2; additive%sample_interval=300
    end select
    write(prefix,'(a,i0)') 'additive',j
    z=additive%fit_transform(x); call matrix('_train',z)
    call matrix('_gram',matmul(z,transpose(z))); call matrix('_query',additive%transform(x+.031_dp))
    call vector('_metadata',real([additive%n_features_in_,size(z,2)],dp))
  end do
  do j=1,6
    skewed=skewed_chi2_sampler(); skewed%n_components=64; skewed%random_state=42+j
    train=x; query=x+.031_dp
    select case(j)
    case(2); skewed%skewedness=.01_dp; skewed%n_components=127
    case(3); skewed%skewedness=0; train=x+.2_dp; query=train+.031_dp
    case(4); skewed%skewedness=-.3_dp; train=x+.5_dp; query=train+.031_dp
    case(5); skewed%skewedness=2; skewed%n_components=1; train=x-.5_dp; query=train-.031_dp
    case(6); skewed%skewedness=.5_dp; skewed%n_components=256; query=x-.1_dp
    end select
    k=skewed%n_components
    if(allocated(u)) deallocate(u,v)
    allocate(u(p,k),v(k)); call seed_rng(skewed%random_state)
    call random_number(u); call random_number(v)
    write(prefix,'(a,i0)') 'skewed',j
    call matrix('_draws_u',u); call vector('_draws_v',v)
    z=skewed%fit_transform(train)
    call matrix('_weights',skewed%random_weights_); call vector('_offset',skewed%random_offset_)
    call matrix('_train',z); call matrix('_query',skewed%transform(query)); call matrix('_gram',matmul(z,transpose(z)))
    call vector('_metadata',real([skewed%n_features_in_,size(z,2)],dp))
  end do
  close(output)
contains
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
  subroutine matrix(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:,:)
    call vector(suffix,reshape(values,[size(values)]))
  end subroutine
end program
