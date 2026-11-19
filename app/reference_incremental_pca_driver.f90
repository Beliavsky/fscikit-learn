program reference_incremental_pca_driver
  use sklearn, only: dp,incremental_pca
  implicit none
  type(incremental_pca) :: model
  real(dp), allocatable :: x(:,:),z(:,:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    model=incremental_pca(); model%n_components=2
    select case(j)
    case(2); model%batch_size=7
    case(3); model%batch_size=9; model%whiten=.true.
    case(4); model%n_components=p; model%batch_size=p+2
    case(5); model%batch_size=8
    case(6); model%n_components=0; model%batch_size=p+2
    case(7); model%batch_size=1; model%n_components=1
    case(8); model%batch_size=11
    end select
    if(j==5) then
      call model%partial_fit(x(:9,:)); call model%partial_fit(x(10:10,:)); call model%partial_fit(x(11:,:))
    else if(j==7) then
      call model%partial_fit(x(:3,:))
      do i=4,n
        call model%partial_fit(x(i:i,:))
      end do
    else
      call model%fit(x)
    end if
    write(prefix,'(a,i0)') 'case',j
    call matrix('_components',model%components_); call vector('_mean',model%mean_); call vector('_var',model%var_)
    call vector('_singular',model%singular_values_); call vector('_variance',model%explained_variance_)
    call vector('_ratio',model%explained_variance_ratio_); call vector('_noise',[model%noise_variance_])
    z=model%transform(x); call matrix('_train',z); call matrix('_inverse',model%inverse_transform(z))
    call matrix('_query',model%transform(x+.137_dp))
    call vector('_shape',real([model%n_samples_seen_,model%n_features_in_,model%n_components_],dp))
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
