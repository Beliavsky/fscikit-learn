program reference_manifold_driver
  use sklearn
  use sklearn_common, only: seed_rng
  implicit none
  type(mds) :: scaling
  type(locally_linear_embedding) :: lle
  real(dp), allocatable :: x(:,:),init(:,:),init3(:,:),d(:,:),missing(:,:),z(:,:),query(:,:),starts(:,:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),init(n,2),init3(n,3),starts(n,2))
  do i=1,n
    read(input,*) x(i,:),init(i,:)
    init3(i,:)=[init(i,:),sin(0.3_dp*i)]
  end do
  close(input); d=pairwise_distances(x,x); missing=d**0.7_dp
  missing(1,2)=0; missing(2,1)=0; missing(3,4)=0; missing(4,3)=0
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    write(prefix,'(a,i0)') 'mds',j
    scaling%metric=j<=3.or.j==8; scaling%normalized_stress='auto'
    scaling%max_iter=300; scaling%eps=1.e-7_dp; scaling%n_init=3; scaling%random_state=42
    scaling%dissimilarity='precomputed'
    if(j==1.or.j==6.or.j==8) scaling%dissimilarity='euclidean'
    if(j==5) scaling%normalized_stress='false'
    if(j==7) then
      scaling%normalized_stress='true'; scaling%max_iter=2
    end if
    select case(j)
    case(1,6); z=scaling%fit_transform(x,init)
    case(2); z=scaling%fit_transform(d,init)
    case(3); scaling%max_iter=1; z=scaling%fit_transform(d,init3)
    case(4,5,7); z=scaling%fit_transform(missing,init)
    case(8)
      call seed_rng(42)
      do i=1,3
        call random_number(starts)
        write(prefix,'(a,i0)') 'start',i; call matrix(trim(prefix),starts)
      end do
      prefix='mds8'; z=scaling%fit_transform(x)
    end select
    call matrix(trim(prefix)//'_embedding',z)
    call matrix(trim(prefix)//'_dissimilarity',scaling%dissimilarity_matrix_)
    call vector(trim(prefix)//'_parameters',[scaling%stress_,real(scaling%n_iter_,dp)])
  end do
  do j=1,4
    write(prefix,'(a,i0)') 'lle',j
    lle%method='standard'; lle%reg=1.e-3_dp; lle%n_neighbors=8; lle%n_components=2
    if(j>=3) lle%method='ltsa'
    if(j==2.or.j==4) then
      lle%n_neighbors=10; lle%n_components=1
    end if
    if(j==3) lle%n_neighbors=16
    if(j==4) lle%n_neighbors=18
    if(j==2) lle%reg=0.05_dp
    z=lle%fit_transform(x); query=lle%transform(x+0.017_dp)
    call matrix(trim(prefix)//'_gram',matmul(z,transpose(z)))
    call matrix(trim(prefix)//'_query_gram',matmul(query,transpose(query)))
    call matrix(trim(prefix)//'_cross_gram',matmul(query,transpose(z)))
    call matrix(trim(prefix)//'_neighbors',real(lle%neighbors_,dp))
    call vector(trim(prefix)//'_error',[lle%reconstruction_error_])
    if(j<=2) call matrix(trim(prefix)//'_weights',lle%weights_)
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
