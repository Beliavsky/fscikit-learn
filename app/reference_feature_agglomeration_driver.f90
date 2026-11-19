program reference_feature_agglomeration_driver
  use sklearn, only: dp,feature_agglomeration
  implicit none
  type(feature_agglomeration) :: model
  real(dp), allocatable :: x(:,:),z(:,:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  character(16), parameter :: links(4)=[character(16)::'ward','single','complete','average']
  character(16), parameter :: pools(3)=[character(16)::'mean','median','max']
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,14
    model=feature_agglomeration(); model%n_clusters=3
    if(j<=12) then
      model%linkage=links((j-1)/3+1); model%pooling_func=pools(mod(j-1,3)+1)
    else if(j==13) then
      model%n_clusters=1
    else
      model%n_clusters=p
    end if
    z=model%fit_transform(x); write(prefix,'(a,i0)') 'case',j
    call vector('_labels',real(model%labels_,dp))
    call matrix('_children',real(reshape([(minval(model%children_(i,:)),i=1,p-1), &
      (maxval(model%children_(i,:)),i=1,p-1)],[p-1,2]),dp))
    call vector('_distances',model%distances_); call matrix('_train',z)
    call matrix('_query',model%transform(x+.137_dp)); call matrix('_inverse',model%inverse_transform(z))
    call vector('_shape',real([model%n_features_in_,model%n_clusters_,model%n_leaves_,model%n_connected_components_],dp))
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
