program reference_lle_variants_driver
  use sklearn, only: dp,locally_linear_embedding
  use sklearn_lle_variants, only: lle_hessian_alignment,lle_modified_alignment,lle_local_spectra
  implicit none
  type(locally_linear_embedding) :: model
  real(dp), allocatable :: x(:,:),query(:,:),z(:,:),transformed(:,:),alignment(:,:),weights(:,:),v(:,:,:),values(:,:)
  integer, allocatable :: sizes(:)
  real(dp) :: eta
  integer :: n,p,k,d,method,unit,out,i
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) n,p,k,d,method,model%hessian_tol,model%modified_tol,model%reg
  allocate(x(n,p),query(5,p))
  do i=1,n
    read(unit,*) x(i,:)
  end do
  do i=1,5
    read(unit,*) query(i,:)
  end do
  close(unit)
  model%n_neighbors=k; model%n_components=d; model%method='hessian'
  if(method==2) model%method='modified'
  z=model%fit_transform(x); transformed=model%transform(query)
  if(method==1) then
    alignment=lle_hessian_alignment(x,model%neighbors_,d,model%hessian_tol)
  else
    alignment=lle_modified_alignment(x,model%neighbors_,d,model%modified_tol,sizes,weights,eta)
    call lle_local_spectra(x,model%neighbors_,v,values)
  end if
  open(newunit=out,file=trim(output),status='replace')
  write(out,'(a,*(i0,1x))') 'neighbors ',transpose(model%neighbors_)
  write(out,'(a,*(es26.17e3,1x))') 'alignment ',transpose(alignment)
  write(out,'(a,*(es26.17e3,1x))') 'embedding ',transpose(z)
  write(out,'(a,*(es26.17e3,1x))') 'query ',transpose(transformed)
  write(out,'(a,*(es26.17e3,1x))') 'error ',model%reconstruction_error_
  write(out,'(a,*(es26.17e3,1x))') 'preserved ',transpose(x),transpose(query)
  if(method==2) then
    write(out,'(a,*(i0,1x))') 'nullity ',sizes
    write(out,'(a,*(es26.17e3,1x))') 'regularized ',transpose(weights)
    write(out,'(a,*(es26.17e3,1x))') 'eta ',eta
    write(out,'(a,*(es26.17e3,1x))') 'spectrum ',transpose(values)
    do i=1,n
      write(out,'(a,i0,1x,*(es26.17e3,1x))') 'basis_',i,transpose(v(i,:,:))
    end do
  end if
  close(out)
end program
