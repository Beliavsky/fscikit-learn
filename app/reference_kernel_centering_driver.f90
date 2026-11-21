program reference_kernel_centering_driver
  use sklearn_kernel_centering
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_negative
  implicit none
  type(kernel_centerer) :: model,copy
  real(dp), allocatable :: kernel(:,:),query(:,:),centered(:,:),result(:,:),original(:,:),signs(:,:)
  character(32), allocatable :: names(:)
  character(1024) :: input,output
  integer :: u,v,n,m,i,j
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=u,file=trim(input),status='old',action='read'); read(u,*) n,m
  allocate(kernel(n,n),query(m,n))
  do i=1,n
    read(u,*) kernel(i,:)
  end do
  do i=1,m
    read(u,*) query(i,:)
  end do
  close(u); original=kernel; centered=model%fit_transform(kernel); result=model%transform(query)
  open(newunit=v,file=trim(output),status='replace',action='write')
  call emit('rows',model%k_fit_rows_); call emit('all',[model%k_fit_all_])
  call emit('train',reshape(centered,[size(centered)])); call emit('query',reshape(result,[size(result)]))
  allocate(signs(n,n)); signs=0
  do j=1,n
    do i=1,n
      if(ieee_is_finite(centered(i,j))) then
        if(centered(i,j)==0.and.ieee_is_negative(centered(i,j))) signs(i,j)=1
      end if
    end do
  end do
  call emit('zero_signs',reshape(signs,[size(signs)]))
  call emit('metadata',[real(model%n_features_in_,dp),merge(1._dp,0._dp,all(kernel==original))])
  names=model%get_feature_names_out(); write(v,'(a,*(1x,a))') 'names',(trim(names(i)),i=1,size(names))
  copy=model
  call model%fit(reshape([2._dp,-3._dp,5._dp,7._dp],[2,2])); result=model%transform(reshape([3._dp,4._dp],[1,2]))
  call emit('refit',reshape(result,[size(result)])); call emit('refit_rows',model%k_fit_rows_)
  call emit('refit_all',[model%k_fit_all_]); call emit('refit_features',[real(model%n_features_in_,dp)])
  result=copy%transform(query); call emit('copy_query',reshape(result,[size(result)])); close(v)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(v,'(a,*(1x,es27.17e3))') name,values
  end subroutine
end program
