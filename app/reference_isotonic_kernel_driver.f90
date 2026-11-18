program reference_isotonic_kernel_driver
  use sklearn
  implicit none
  type(isotonic_regression) :: iso
  type(rbf_sampler) :: rbf
  type(nystroem) :: ny
  real(dp), allocatable :: x(:,:),y(:),w(:),t(:),q(:),z(:),features(:,:),ones(:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  character(16) :: kernels(5)=[character(16)::'rbf','rbf','polynomial','linear','cosine']
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n),w(n),t(n),q(n+2),ones(n))
  do i=1,n
    read(input,*) x(i,:),y(i),w(i)
  end do
  close(input); t=x(:,1); q(:n)=t+0.17_dp; q(n+1)=minval(t)-2; q(n+2)=maxval(t)+2; ones=1
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,6
    write(prefix,'(a,i0)') 'iso',j
    iso%increasing=j/=2; iso%auto_increasing=j==3
    iso%y_min=-huge(1._dp); iso%y_max=huge(1._dp); iso%out_of_bounds='clip'
    if(j==3) then
      iso%y_min=-1; iso%y_max=2
    end if
    if(j==4) iso%out_of_bounds='nan'
    select case(j)
    case(1); z=iso%fit_transform(t,y)
    case(5); z=iso%fit_transform(ones,y,w)
    case(6)
      ones=0; ones(1)=1; z=iso%fit_transform(t,y,ones)
    case default; z=iso%fit_transform(t,y,w)
    end select
    call vector(trim(prefix)//'_fit',z); call vector(trim(prefix)//'_query',iso%predict(q))
    call vector(trim(prefix)//'_x',iso%x_thresholds_); call vector(trim(prefix)//'_y',iso%y_thresholds_)
    call vector(trim(prefix)//'_parameters',[iso%x_min_,iso%x_max_,merge(1._dp,0._dp,iso%increasing_)])
  end do
  call vector('pava_default',isotonic_fit(y))
  call vector('pava_weighted',isotonic_fit(y,w+0.1_dp))
  call vector('pava_decreasing',isotonic_fit(y,w+0.1_dp,increasing=.false.))
  call vector('pava_bounded',isotonic_fit(y,w+0.1_dp,y_min=-0.5_dp,y_max=1.5_dp))
  do j=1,3
    write(prefix,'(a,i0)') 'rbf',j
    rbf%gamma=0.7_dp; rbf%gamma_scale=j==3; rbf%n_components=150
    if(j==2) rbf%gamma=0
    rbf%random_state=42+j; features=rbf%fit_transform(x)
    call matrix(trim(prefix)//'_weights',rbf%random_weights_)
    call vector(trim(prefix)//'_offset',rbf%random_offset_)
    call vector(trim(prefix)//'_gamma',[rbf%gamma_]); call matrix(trim(prefix)//'_fit',features)
    call matrix(trim(prefix)//'_query',rbf%transform(x+0.031_dp))
  end do
  do j=1,5
    write(prefix,'(a,i0)') 'ny',j
    ny%kernel=kernels(j); ny%gamma=-1; ny%n_components=6; ny%degree=2; ny%coef0=0.7_dp
    if(j==2) ny%gamma=0.7_dp
    if(j>=4) ny%n_components=p-1
    ny%random_state=42+j; features=ny%fit_transform(x)
    call vector(trim(prefix)//'_indices',real(ny%component_indices_,dp))
    call matrix(trim(prefix)//'_components',ny%components_); call matrix(trim(prefix)//'_normalization',ny%normalization_)
    call matrix(trim(prefix)//'_fit',features); call matrix(trim(prefix)//'_query',ny%transform(x+0.031_dp))
  end do
  close(output)
contains
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,values
  end subroutine
  subroutine matrix(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:,:)
    call vector(name,reshape(values,[size(values)]))
  end subroutine
end program
